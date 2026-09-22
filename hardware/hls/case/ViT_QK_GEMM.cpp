#include "../src/common.h"
#include "../src/vit_qk_gemm.h"
#include "../src/quantizer.h"
#include "../src/utils.h"

// ViT QK_GEMM 仿真：双向注意力，无 KV Cache，无因果掩码，无 ROPE
// 数据流：ViT_DEMUX → qk_stream（Q/K 按 head 交错，量化前）→ QUANTIZER → VIT_QK_GEMM → R
//
// 与 LLM_QK_GEMM 的区别：
//   LLM  : 需要 ROPE，使用 CONDENSED_ROPE_QK_QUANT.bin，有 KV Cache，T=8（自回归）
//   ViT  : 无 ROPE，使用 CONDENSED_DEMUX_QK.bin，先量化再做 QK_GEMM，无 Cache，T=1024（双向）
//
// 参数简化说明（去掉原 T_TILE）：
//   ViT QK_GEMM 必须在内部缓存全部 K tokens（S=1024），Q 以 TP=1 逐 token 处理，
//   T_TILE 没有意义——硬件 tile 大小就是 TP=1，外层循环直接由模块内部 T 决定。

// ── 仿真超参数 ──────────────────────────────────────────────────────────────
constexpr int L      = VIT_L;    // 12

// ── 模型超参数 ──────────────────────────────────────────────────────────────
constexpr int H      = VIT_H;    // 12
constexpr int G      = VIT_G;    // 8
constexpr int HC     = VIT_HC;   // 64

// ── 设计超参数（精简后只保留 T、TP 两个 token 维度参数）──────────────────────
constexpr int T      = 1024;    // 1024：总 token 数（Q 和 K 均为 T）
constexpr int TP     = 1;        // Q token 并行度：每拍处理 TP=1 个 Q token
constexpr int S      = VIT_S;    // 1024：K 序列长度（ViT 中 S == T）
constexpr int SP     = G;        // 8：K 并行度（输出 R 每拍 SP=8 个 K 位置）
constexpr int CP     = G;        // 8：channel 并行度

// ── 派生超参数 ──────────────────────────────────────────────────────────────
constexpr int ST     = S  / SP;  // 128：K tiles
constexpr int HCT    = HC / CP;  // 8：head-channel tiles

// ── 数据类型 ─────────────────────────────────────────────────────────────────
constexpr int TRUNC  = VIT_MHA_TRUNC_R;
typedef ap_int <VIT_DW_AQ      > aq_t;   // int8  激活
typedef ap_uint<VIT_DW_AS      > as_t;   // uint4 scale
typedef ap_int <VIT_DW_R       > acc_t;  // 40-bit 累加
typedef ap_int <VIT_DW_R_TRUNC > of_t;   // 31-bit 截断
typedef VIT_QKV_TRUNC_T iq_t;            // QKV trunc 后、量化前输入

// ── 元素数量 ─────────────────────────────────────────────────────────────────
// qk_stream 来自 DEMUX，格式 [h][qk=Q/K][tt][hct][t]×CP，Q 和 K 交错
// NUM_QK = H * 2 * T * HC（Q+K 激活总量）
// NUM_QK_S = H * 2 * T * HCT（Q+K scale 总量，每 hct 组一个 scale）
constexpr int NUM_QK   = 2 * H * T * HC;
constexpr int NUM_R    =     H * T * S;   // R[H, T, S] 注意力得分总量

// ── 模块实例化 ───────────────────────────────────────────────────────────────
// 去掉 T_TILE 参数：模块以 T=1024、TP=1 在内部处理全部 Q tokens（无外层 tile 循环）
VIT_QK_GEMM<aq_t, as_t, acc_t, of_t, TRUNC, H, T, TP, S, SP, HC, CP> vit_qk_gemm_inst;
QUANTIZER<iq_t, aq_t, as_t, 1, 2 * H * T, 1, HC, CP, G> vit_qk_quantizer_inst;


// ============================================================================
// top
// ============================================================================
void top(
    int l_begin,
    int l_close,
    // 单路交错 QK 流（量化前），顺序 [h][qk=Q/K][t][hct]
    hls::stream<hls::vector<iq_t, CP    > >& qk_i_stream,
    hls::stream<hls::vector<of_t, TP*SP> >& r_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=qk_i_stream
    #pragma HLS interface axis port=r_stream
    #pragma HLS aggregate variable=qk_i_stream compact=bit
    #pragma HLS aggregate variable=r_stream    compact=bit

    for(int l = l_begin; l < l_close && l < VIT_L; ++l){
        hls::stream<hls::vector<aq_t, CP> > qk_q_stream("qk_q_stream");
        hls::stream<hls::vector<as_t, 1 > > qk_s_stream("qk_s_stream");

        // 无 ROPE：直接对 DEMUX 的 QK 交错输出做动态量化，交错流直接送入 VIT_QK_GEMM
        vit_qk_quantizer_inst.do_quant(qk_i_stream, qk_q_stream, qk_s_stream);

        vit_qk_gemm_inst.do_vit_qk_gemm(qk_q_stream, qk_s_stream, r_stream);
    }
}


// ============================================================================
// 参考数据声明
// ============================================================================
// CONDENSED_DEMUX_QK.bin 内容（激活，类型 VIT_QKV_TRUNC_T，存为 int64_t）
int64_t REF_CONDENSED_QK_Q[NUM_QK  ];   // 从 bin 文件加载的激活（量化前）

// 期望输出
int64_t REF_R[H * T * S];
int64_t DUT_R[H * T * S];


// ============================================================================
// test_layer
// ============================================================================
void test_layer(int l){
    string file_path = VIT_BINARIES_PATH + to_string(l);
    string save_path = VIT_CONDENSE_PATH + to_string(l);

    // ── 加载参考数据 ──────────────────────────────────────────────────────────
    // MHA_R.bin 由软件以 [B=5, H=12, T=1024, S=1024] 顺序保存（非 [H, B*T, S]）
    // 正确读法：把文件视为 [B_LOAD*H=60][T=1024][S=1024]，只取前 H=12 个"头"
    // 即 image-0 的全部 12 个 attention 头（前 H*T*S = 12582912 个元素）
    constexpr int B_LOAD = 5;
    {
        auto R   = read_tensor<int64_t>(file_path + "/MHA_R.bin"  );

        tensor2array<int64_t>(R,   REF_R,   B_LOAD*H, H, T, 0, T, S,   S   );
    }

    // ── 从 CONDENSED_DEMUX_QK.bin 加载交错激活 ───────────────────────────────
    // 文件由 ViT_DEMUX testbench 保存，格式与 LLM 的 CONDENSED_ROPE_QK_QUANT 等价：
    //   内容：save_condensed_tensor<int64_t, VIT_QKV_TRUNC_T, 2*T*C, CP>(...)
    //   流顺序：[h][qk=Q/K][tt][hct][t] × CP=8，共 NUM_QK/CP 个向量
    {
        auto CONDENSED_QK = read_tensor<int64_t>(save_path + "/CONDENSED_DEMUX_QK.bin");
        tensor2array<int64_t>(CONDENSED_QK, REF_CONDENSED_QK_Q, 1, 1, 1, 1, NUM_QK, NUM_QK);
    }

    // ── 创建 HLS 流 ───────────────────────────────────────────────────────────
    hls::stream<hls::vector<iq_t,   CP > > qk_i_stream("qk_i_stream");
    hls::stream<hls::vector<of_t,TP*SP > > r_stream   ("r_stream"   );

    // 量化前输入：从 CONDENSED_DEMUX_QK.bin 加载（int64_t → VIT_QKV_TRUNC_T）
    array2stream<int64_t, iq_t, 1, 1, 1, 1, NUM_QK, CP>(REF_CONDENSED_QK_Q, qk_i_stream, "QK_I", true);

    // ── 调用 top ──────────────────────────────────────────────────────────────
    top(l, l+1, qk_i_stream, r_stream);

    // ── 保存 condensed 输出（供链式仿真使用）────────────────────────────────────
    save_condensed_tensor<int64_t, of_t, NUM_R, TP*SP>(
        save_path + "/CONDENSED_QK_GEMM_R.bin", r_stream);

    // ── 读取并对比输出 ────────────────────────────────────────────────────────
    // r_stream 顺序：[h][t_q][st] × SP，其中 st = s/SP，即 [H][T][S/SP][SP]
    stream2array<int64_t, of_t, H, T, TP, S, SP>(r_stream, DUT_R, "Output R", true);

    assert(qk_i_stream.empty());
    assert(r_stream   .empty());

    // 双向注意力：所有 (h, t, s) 位置均有效，直接全量对比
    int mismatch = 0;
    for(int h = 0; h < H; ++h){
        for(int t = 0; t < T; ++t){
            for(int s = 0; s < S; ++s){
                int idx = h*T*S + t*S + s;
                if(REF_R[idx] != DUT_R[idx]){
                    if(++mismatch <= 10)
                        printf("R mismatch L%d H%d t%d s%d: REF=%ld DUT=%ld\n",
                               l, h, t, s, REF_R[idx], DUT_R[idx]);
                }
            }
        }
    }
    if(mismatch > 0) printf("Layer %d: R total mismatch = %d\n", l, mismatch);
    else             printf("Layer %d: R PASS\n", l);
}


int main(){
    for(int l = 0; l < 1; ++l){
        test_layer(l);
        printf("Layer %d done\n", l);
    }
    return 0;
}
