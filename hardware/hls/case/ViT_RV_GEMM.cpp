#include "../src/common.h"
#include "../src/vit_rv_gemm.h"
#include "../src/utils.h"

// ViT RV_GEMM 仿真：双向注意力，无外部 V Cache
// 输入：R（Softmax 量化后）[H][S_VIT][S_VIT] + V 原始 [H][S_VIT][HC]
// 输出：A（注意力加权聚合）[H][S_VIT][HC]

// ── 仿真超参数 ──────────────────────────────────────────────────
constexpr int L      = VIT_L;   // 12

// ── 模型超参数 ──────────────────────────────────────────────────
constexpr int H      = VIT_H;   // 12
constexpr int G      = VIT_G;   // 8
constexpr int HC     = VIT_HC;  // 64

// ── 设计超参数 ──────────────────────────────────────────────────
// T_TILE = mini-batch 大小（硬件 tile），T = 总 token 数，S = S_VIT（全序列）
// 内部循环 ST_VIT = S/T_TILE = 128 次，每次处理 T_TILE 行 R
constexpr int T_TILE = 8;       // R mini-batch 大小（硬件 tile，原 T=8）
constexpr int T      = VIT_S;   // 1024（总 token 数）
constexpr int TP     = 1;
constexpr int S      = VIT_S;   // 1024
constexpr int SP     = G;       // 8
constexpr int CP     = G;       // 8

// ── 派生超参数 ──────────────────────────────────────────────────
constexpr int TT     = T_TILE / TP;
constexpr int ST     = S     / SP;
constexpr int HCT    = HC    / CP;
constexpr int ST_VIT = S     / T_TILE;  // 128 个 mini-batch

constexpr int TRUNC  = VIT_MHA_TRUNC_A;  // 17

// ── 数据类型 ────────────────────────────────────────────────────
typedef ap_int <VIT_DW_AQ       > aq_t;    // 8-bit 激活量化（R & V）
typedef ap_uint<VIT_DW_AS       > as_t;    // 4-bit scale
typedef ap_int <VIT_DW_QKV_TRUNC> w_t;    // 20-bit V 原始（截断后）
typedef ap_int <VIT_DW_A        > acc_t;   // 40-bit 累加
typedef ap_int <VIT_DW_A_TRUNC  > of_t;   // 23-bit 截断后

// ── 元素数量 ─────────────────────────────────────────────────────
constexpr int NUM_V  = H * S   * HC;  // V 总元素数
constexpr int NUM_R  = H * S   * S;   // R 总元素数（H*S_VIT^2）
constexpr int NUM_A  = H * S   * HC;  // A 总元素数

// ── 模块实例化 ───────────────────────────────────────────────────
VIT_RV_GEMM<aq_t, as_t, w_t, acc_t, of_t, TRUNC, H, T_TILE, TP, S, SP, HC, CP> vit_rv_gemm_inst;

// 输出量化器（A → 量化 int8）
QUANTIZER<of_t, aq_t, as_t, H, S, TP, HC, CP, G> quantizer_inst;
Adapter  <of_t,   H*S*HCT, 1, TP*CP, TP*CP, CP> adapter_1;
Adapter  <aq_t,   H*S*HCT, 1, TP*CP,    CP, TP*CP> adapter_2;
Adapter  <as_t,   H*S*HCT, 1, TP,        1, TP   > adapter_3;


void top(
    int l_begin,
    int l_close,
    hls::stream<hls::vector<aq_t, TP*SP> >& rq_stream,
    hls::stream<hls::vector<as_t, TP   > >& rs_stream,
    hls::stream<hls::vector<w_t,     CP> >& v_stream,
    hls::stream<hls::vector<aq_t, TP*CP> >& aq_stream,
    hls::stream<hls::vector<as_t, TP   > >& as_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=rq_stream
    #pragma HLS interface axis port=rs_stream
    #pragma HLS interface axis port=v_stream
    #pragma HLS interface axis port=aq_stream
    #pragma HLS interface axis port=as_stream

    #pragma HLS aggregate variable=rq_stream compact=bit
    #pragma HLS aggregate variable=rs_stream compact=bit
    #pragma HLS aggregate variable=v_stream  compact=bit
    #pragma HLS aggregate variable=aq_stream compact=bit
    #pragma HLS aggregate variable=as_stream compact=bit

    hls::stream<hls::vector<of_t, TP*CP> > a_stream      ("a_stream"      );
    hls::stream<hls::vector<of_t,    CP> > a_stream_adpt ("a_stream_adpt" );
    hls::stream<hls::vector<aq_t,    CP> > aq_stream_adpt("aq_stream_adpt");
    hls::stream<hls::vector<as_t,     1> > as_stream_adpt("as_stream_adpt");

    for(int l=l_begin; l<l_close && l<VIT_L; ++l){
        #pragma HLS dataflow

        vit_rv_gemm_inst.do_vit_rv_gemm(rq_stream, rs_stream, v_stream, a_stream);

        adapter_1     .do_adapt(a_stream,        a_stream_adpt );
        quantizer_inst.do_quant(a_stream_adpt,   aq_stream_adpt, as_stream_adpt);
        adapter_2     .do_adapt(aq_stream_adpt,  aq_stream     );
        adapter_3     .do_adapt(as_stream_adpt,  as_stream     );
    }
}


// ── 参考数据 ────────────────────────────────────────────────────
int8_t  REF_R_Q [H * S * S ];
int8_t  REF_R_S [H * S * ST];
int64_t REF_V   [H * S * HC];
int8_t  REF_A_Q [H * S * HC];
int8_t  REF_A_S [H * S * HCT];
int8_t  DUT_A_Q [H * S * HC];
int8_t  DUT_A_S [H * S * HCT];

int8_t  REF_COND_RQ[H * S * S ];
int8_t  REF_COND_RS[H * S * ST];
int64_t REF_COND_V [H * S * HC];


void test_layer(int l){
    string file_path = VIT_BINARIES_PATH + to_string(l);
    string save_path = VIT_CONDENSE_PATH + to_string(l);

    // 所有 binary 文件均以 batch-first 顺序保存：
    //   MHA_R_Q/R_S.bin: [B=5, H=12, T=1024, S/S_G]，视为 [B*H, T, S/S_G]
    //   MHA_V_SPLIT_HEADS.bin: [B=5, H=12, T=1024, HC]，视为 [B*H, T, HC]
    //   MHA_A_Q/A_S.bin: [B=5, T=1024, H*HC/H*HCT]（a_gen 在转置前已 reshape），
    //                     视为 [1, B*T, H*HC/H*HCT]，只取前 T=1024 行（image-0）
    constexpr int B_LOAD = 5;
    {
        auto R_Q = read_tensor<int8_t >(file_path + "/MHA_R_Q.bin");
        auto R_S = read_tensor<int8_t >(file_path + "/MHA_R_S.bin");
        auto V   = read_tensor<int64_t>(file_path + "/MHA_V_SPLIT_HEADS.bin");
        auto A_Q = read_tensor<int8_t >(file_path + "/MHA_A_Q.bin");
        auto A_S = read_tensor<int8_t >(file_path + "/MHA_A_S.bin");

        tensor2array<int8_t >(R_Q, REF_R_Q, B_LOAD*H, H, S, 0, S, S,   S  );
        tensor2array<int8_t >(R_S, REF_R_S, B_LOAD*H, H, S, 0, S, ST,  ST );
        tensor2array<int64_t>(V,   REF_V,   B_LOAD*H, H, S, 0, S, HC,  HC );
        tensor2array<int8_t >(A_Q, REF_A_Q, 1, 1, B_LOAD*S, 0, S, H*HC,  H*HC );
        tensor2array<int8_t >(A_S, REF_A_S, 1, 1, B_LOAD*S, 0, S, H*HCT, H*HCT);
    }
    split_heads<int8_t>(REF_A_Q, H, S, HC );
    split_heads<int8_t>(REF_A_S, H, S, HCT);

    // 使用 condense 输入
    auto COND_RQ = read_tensor<int8_t >(save_path + "/CONDENSED_SOFTMAX_R_Q.bin");
    auto COND_RS = read_tensor<int8_t >(save_path + "/CONDENSED_SOFTMAX_R_S.bin");
    auto COND_V  = read_tensor<int64_t>(save_path + "/CONDENSED_DEMUX_V.bin");

    tensor2array<int8_t >(COND_RQ, REF_COND_RQ, 1, 1, 1, 1, H*S*S,   H*S*S  );
    tensor2array<int8_t >(COND_RS, REF_COND_RS, 1, 1, 1, 1, H*S*ST,  H*S*ST );
    tensor2array<int64_t>(COND_V,  REF_COND_V,  1, 1, 1, 1, H*S*HC,  H*S*HC );

    // 创建流
    hls::stream<hls::vector<aq_t, TP*SP> > rq_stream("rq_stream");
    hls::stream<hls::vector<as_t, TP   > > rs_stream("rs_stream");
    hls::stream<hls::vector<w_t,     CP> > v_stream ("v_stream" );
    hls::stream<hls::vector<aq_t, TP*CP> > aq_stream("aq_stream");
    hls::stream<hls::vector<as_t, TP   > > as_stream("as_stream");

    // 填入流（R 按 condensed 顺序，V 按 DEMUX 格式 [H][TT_D][CT][TP_D]×CP）
    array2stream<int8_t, aq_t, 1, 1, 1, 1, H*S*S,  TP*SP>(REF_COND_RQ, rq_stream, "RQ", true);
    array2stream<int8_t, as_t, 1, 1, 1, 1, H*S*ST, TP   >(REF_COND_RS, rs_stream, "RS", true);
    array2stream<int64_t,w_t,  1, 1, 1, 1, H*S*HC, CP   >(REF_COND_V,  v_stream,  "V",  true);

    top(l, l+1, rq_stream, rs_stream, v_stream, aq_stream, as_stream);

    save_condensed_tensor<int8_t, aq_t, H*S*HC,  TP*CP>(save_path + "/CONDENSED_RV_GEMM_A_Q.bin", aq_stream);
    save_condensed_tensor<int8_t, as_t, H*S*HCT, TP   >(save_path + "/CONDENSED_RV_GEMM_A_S.bin", as_stream);

    stream2array<int8_t, aq_t, H, S, TP, HC,  CP>(aq_stream, DUT_A_Q, "A_Q", true);
    stream2array<int8_t, as_t, H, S, TP, HCT,  1>(as_stream, DUT_A_S, "A_S", true);

    assert(rq_stream.empty());
    assert(rs_stream.empty());
    assert(v_stream .empty());
    assert(aq_stream.empty());
    assert(as_stream.empty());

    compare<int8_t>(REF_A_Q, DUT_A_Q, H*S*HC,  "A_Q", true);
    compare<int8_t>(REF_A_S, DUT_A_S, H*S*HCT, "A_S", true);
    printf("Layer %d done\n", l);
}


int main(){
    for(int l=0; l<1; ++l){
        test_layer(l);
    }
    return 0;
}
