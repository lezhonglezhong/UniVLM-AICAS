#include "../src/common.h"
#include "../src/vit_softmax.h"
#include "../src/quantizer.h"
#include "../src/adapter.h"

// ViT Softmax 仿真：双向注意力，无因果掩码
// 输入：QK_GEMM 截断后的注意力得分 R[H][S_VIT][S_VIT]
// 输出：Softmax 权重（量化为 int8）+ scale

// ── 仿真超参数 ──────────────────────────────────────────────────
constexpr int L      = VIT_L;   // 12

// ── 模型超参数 ──────────────────────────────────────────────────
constexpr int H      = VIT_H;   // 12
constexpr int G      = VIT_G;   // 8

// ── 设计超参数 ──────────────────────────────────────────────────
constexpr int T      = VIT_S;   // 1024（= S_VIT，全序列）
constexpr int TP     = 1;
constexpr int S      = VIT_S;   // 1024
constexpr int SP     = G;       // 8
constexpr int SAP    = G;       // SAP = SAP（Adapter packing，与 SP 一致）

// ── 派生超参数 ──────────────────────────────────────────────────
constexpr int TT     = T  / TP;           // 1024
constexpr int ST     = S  / SAP;          // 128
constexpr int NUM_R  = H * T * S;
constexpr int NUM_RS = H * T * S / G;

// ── 数据类型 ────────────────────────────────────────────────────
// 直接使用 common.h 中已定义的 VIT_* 类型，避免与 LLM 同名 typedef 冲突
// VIT_R_TRUNC_T  = ap_int<31>，VIT_R_MASKED_T = ap_int<31>
// VIT_EXP_T      = ap_int<9>，  VIT_EXP_SUM_T  = ap_int<19>
// VIT_RECIP_T    = ap_int<18>， VIT_SOFTMAX_T   = ap_int<23>

// ── 模块实例化 ───────────────────────────────────────────────────
VIT_SOFTMAX<VIT_R_TRUNC_T, VIT_R_MASKED_T, VIT_EXP_T, VIT_EXP_SUM_T, VIT_RECIP_T, VIT_SOFTMAX_T,
            H, T, TP, S, SAP, SP> vit_softmax_inst;

QUANTIZER  <VIT_SOFTMAX_T, AQ_T, AS_T, H, T, TP, S, SAP, G> quantizer_inst;


void top(
    int l_begin,
    int l_close,
    hls::stream<hls::vector<VIT_R_TRUNC_T,  TP*SAP> >& r_stream,
    hls::stream<hls::vector<AQ_T,           TP*SAP> >& rq_stream,
    hls::stream<hls::vector<AS_T,           TP     > >& rs_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=r_stream
    #pragma HLS interface axis port=rq_stream
    #pragma HLS interface axis port=rs_stream

    #pragma HLS aggregate variable=r_stream  compact=bit
    #pragma HLS aggregate variable=rq_stream compact=bit
    #pragma HLS aggregate variable=rs_stream compact=bit

    hls::stream<hls::vector<VIT_SOFTMAX_T, TP*SAP> > softmax_stream;

    for(int l=l_begin; l<l_close && l<VIT_L; ++l){
        #pragma HLS dataflow
        vit_softmax_inst.do_vit_softmax(r_stream, softmax_stream);
        quantizer_inst  .do_quant      (softmax_stream, rq_stream, rs_stream);
    }
}


// ── 参考数据 ────────────────────────────────────────────────────
int64_t REF_R   [H * T * S];
int8_t  REF_R_Q [H * T * S];
int8_t  REF_R_S [H * T * ST];
int8_t  DUT_R_Q [H * T * S];
int8_t  DUT_R_S [H * T * ST];
int64_t REF_CONDENSED_R[H * T * S];

#define COMPARE_R_Q 1
#define COMPARE_R_S 1


void test_layer(int l){
    string file_path = VIT_BINARIES_PATH + to_string(l);
    string save_path = VIT_CONDENSE_PATH + to_string(l);

    // MHA_R.bin / MHA_R_Q.bin / MHA_R_S.bin 均以 [B=5, H=12, T=1024, ...] 顺序保存
    // 需把文件视为 [B_LOAD*H, T, ...] 才能通过 size check，只取前 H 个头（image-0）
    constexpr int B_LOAD = 5;
    {
        auto R   = read_tensor<int64_t>(file_path + "/MHA_R.bin"  );
        auto R_Q = read_tensor<int8_t >(file_path + "/MHA_R_Q.bin");
        auto R_S = read_tensor<int8_t >(file_path + "/MHA_R_S.bin");

        tensor2array<int64_t>(R,   REF_R,   B_LOAD*H, H, T, 0, T, S,  S );
        tensor2array<int8_t >(R_Q, REF_R_Q, B_LOAD*H, H, T, 0, T, S,  S );
        tensor2array<int8_t >(R_S, REF_R_S, B_LOAD*H, H, T, 0, T, ST, ST);
    }

    // 使用 condense 输入（来自 ViT QK_GEMM 链式输出）
    auto CONDENSED_R = read_tensor<int64_t>(save_path + "/CONDENSED_QK_GEMM_R.bin");
    tensor2array<int64_t>(CONDENSED_R, REF_CONDENSED_R, 1, 1, 1, 1, NUM_R, NUM_R);

    hls::stream<hls::vector<VIT_R_TRUNC_T, TP*SAP> > r_stream ("r_stream" );
    hls::stream<hls::vector<AQ_T,          TP*SAP> > rq_stream("rq_stream");
    hls::stream<hls::vector<AS_T,          TP     > > rs_stream("rs_stream");

    array2stream<int64_t, VIT_R_TRUNC_T, 1, 1, 1, 1, NUM_R, TP*SAP>(REF_CONDENSED_R, r_stream, "R", true);

    top(l, l+1, r_stream, rq_stream, rs_stream);

    save_condensed_tensor<int8_t, AQ_T, NUM_R,   TP*SAP>(save_path + "/CONDENSED_SOFTMAX_R_Q.bin", rq_stream);
    save_condensed_tensor<int8_t, AS_T, NUM_RS,  TP    >(save_path + "/CONDENSED_SOFTMAX_R_S.bin", rs_stream);

    stream2array<int8_t, AQ_T, H, T, TP, S,  SAP>(rq_stream, DUT_R_Q, "R_Q", true);
    stream2array<int8_t, AS_T, H, T, TP, ST,   1>(rs_stream, DUT_R_S, "R_S", true);

    assert(r_stream .empty());
    assert(rq_stream.empty());
    assert(rs_stream.empty());

    int mm_rq = 0, mm_rs = 0;
#if COMPARE_R_Q
    for(int h=0; h<H; ++h)
        for(int t=0; t<T; ++t)
            for(int s=0; s<S; ++s){
                int idx = h*T*S + t*S + s;
                if(REF_R_Q[idx] != DUT_R_Q[idx]){
                    mm_rq++;
                    if(mm_rq <= 5) printf("R_Q mismatch H%d T%d S%d: REF=%d DUT=%d\n",
                                          h,t,s,(int)REF_R_Q[idx],(int)DUT_R_Q[idx]);
                }
            }
#endif
#if COMPARE_R_S
    for(int h=0; h<H; ++h)
        for(int t=0; t<T; ++t)
            for(int st=0; st<ST; ++st){
                int idx = h*T*ST + t*ST + st;
                if(REF_R_S[idx] != DUT_R_S[idx]){
                    mm_rs++;
                    if(mm_rs <= 5) printf("R_S mismatch H%d T%d ST%d: REF=%d DUT=%d\n",
                                          h,t,st,(int)REF_R_S[idx],(int)DUT_R_S[idx]);
                }
            }
#endif
    printf("Layer %d: R_Q mismatch=%d  R_S mismatch=%d\n", l, mm_rq, mm_rs);
}


int main(){
    for(int l=0; l<1; ++l){
        test_layer(l);
        printf("Layer %d done\n", l);
    }
    return 0;
}
