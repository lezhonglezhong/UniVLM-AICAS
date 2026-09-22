#include "../src/common.h"
#include "../src/gelu.h"
#include "../src/quantizer.h"
#include "../src/utils.h"

// ============================================================================
// ViT_GELU：ViT MLP 的 GELU 激活 + 动态量化（对应 LLM_SILU_EM）
// ============================================================================
// 与 LLM SwiGLU 的关键区别：ViT MLP 只有单路 fc1→GELU→fc2，无 gate 乘法。
// 输入：CONDENSED_DEMUX_FC1.bin（来自 ViT_DEMUX）
//       数据流顺序：TT -> CT -> TP x CP（流接口为 unpacked，每拍 CP）
// 输出：CONDENSED_GELU_XM_Q.bin（AQ_T，int8）
//       CONDENSED_GELU_XM_S.bin（AS_T，uint4）
// 比对：MLP_XM_Q.bin / MLP_XM_S.bin（VIT_BINARIES_PATH，B_LOAD×T 行）
// ============================================================================

// simulation hyperparameters
constexpr int L         = VIT_L;     // 12
constexpr int T_LOAD    = 5120;      // saved seq length (batch * patches)
constexpr int POS       = 0;         // start position (取第一张图的 1024 patches)

// model hyperparameters
constexpr int C         = VIT_MLP_DIM;  // 3072
constexpr int G         = VIT_G;        // 8

// design hyperparameters (TP=1: 除 MUX/RV_GEMM 外其他模块每拍处理 1 token)
constexpr int T         = VIT_S;    // 1024
constexpr int TP        = 1;
constexpr int CP        = G;        // 8

// derived hyperparameters
constexpr int TT        = T / TP;   // 1024
constexpr int CT        = C / CP;   // 384 (= CMT)
constexpr int CG        = C / G;    // 384，每个 token 的 scale 数
constexpr int CPG       = CP / G;   // 1
// DEMUX fc1_stream 的 token 并行度（TT=128→CMT=384→DEMUX_TP=8 顺序）
// stream2array_unpack 需用此值正确还原 (token, channel) 布局
constexpr int DEMUX_TP  = 8;

// instantiate GELU and Quantizer
GELU     <VIT_FC1_TRUNC_T, VIT_FC1_GELU_T, T, TP, C, CP         > gelu_inst;
QUANTIZER<VIT_FC1_GELU_T,  AQ_T, AS_T, 1,  T, TP, C, CP, G      > quantizer_inst;

// top function
void top(
    int l_begin,
    int l_close,
    hls::stream<hls::vector<VIT_FC1_TRUNC_T, CP   > > &fc1_stream,
    hls::stream<hls::vector<AQ_T,            CP   > > &q_stream,
    hls::stream<hls::vector<AS_T,            TP*CPG> > &s_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=fc1_stream
    #pragma HLS interface axis port=q_stream
    #pragma HLS interface axis port=s_stream

    #pragma HLS aggregate variable=fc1_stream compact=bit
    #pragma HLS aggregate variable=q_stream   compact=bit
    #pragma HLS aggregate variable=s_stream   compact=bit

    // 中间流：GELU 输出 → Quantizer 输入
    hls::stream<hls::vector<VIT_FC1_GELU_T, CP> > gelu_stream("gelu_stream");

    for(int l = l_begin; l < l_close && l < VIT_L; ++l){
        #pragma HLS dataflow
        gelu_inst.do_gelu_unpack  (fc1_stream,  gelu_stream            );
        quantizer_inst.do_quant        (gelu_stream, q_stream, s_stream);
    }
}

// declare ref data
int64_t REF_FC1 [T*C ];
int8_t  REF_XM_Q[T*C ];
int8_t  REF_XM_S[T*CG];
// declare dut data
int8_t  DUT_XM_Q[T*C ];
int8_t  DUT_XM_S[T*CG];

void test_layer(int l){
    string file_path = VIT_BINARIES_PATH + to_string(l);
    string save_path = VIT_CONDENSE_PATH + to_string(l);

    //* read reference data
    {
        auto REF_XM_Q_TENSOR = read_tensor<int8_t>(file_path + "/MLP_XM_Q.bin");
        auto REF_XM_S_TENSOR = read_tensor<int8_t>(file_path + "/MLP_XM_S.bin");
        assert(REF_XM_Q_TENSOR.size() == (size_t)(T_LOAD * C ));
        assert(REF_XM_S_TENSOR.size() == (size_t)(T_LOAD * CG));
        //            dtype,   tensor,         array,     H_LOAD, H, T_LOAD, T_START, T,  C_LOAD, C
        tensor2array<int8_t>(REF_XM_Q_TENSOR, REF_XM_Q,  1, 1,  T_LOAD, POS, T,  C,  C );
        tensor2array<int8_t>(REF_XM_S_TENSOR, REF_XM_S,  1, 1,  T_LOAD, POS, T,  CG, CG);
    }

    //* create streams
    hls::stream<hls::vector<VIT_FC1_TRUNC_T, CP   > > fc1_stream("fc1_stream");
    hls::stream<hls::vector<AQ_T,            CP   > > q_stream  ("q_stream"  );
    hls::stream<hls::vector<AS_T,            TP*CPG> > s_stream ("s_stream"  );

    //* use condensed input directly (already TT -> CT -> TP x CP)
    constexpr int NUM_FC1 = T * C;  // 3,145,728
    auto CONDENSED_FC1 = read_tensor<int64_t>(save_path + "/CONDENSED_DEMUX_FC1.bin");
    tensor2array<int64_t>(CONDENSED_FC1, REF_FC1, 1, 1, 1, 1, NUM_FC1, NUM_FC1);
    array2stream<int64_t, VIT_FC1_TRUNC_T, 1, 1, 1, 1, NUM_FC1, CP>(REF_FC1, fc1_stream, "CONDENSED FC1", true);

    //* call top function
    top(l, l+1, fc1_stream, q_stream, s_stream);

    //* save condensed output
    save_condensed_tensor<int8_t, AQ_T, T*C,  CP    >(save_path + "/CONDENSED_GELU_XM_Q.bin", q_stream);
    save_condensed_tensor<int8_t, AS_T, T*CG, TP*CPG>(save_path + "/CONDENSED_GELU_XM_S.bin", s_stream);

    //* read output from stream
    // DEMUX fc1_stream 顺序为 TT=128→CMT=384→TP=8，GELU/Quant 保持同序输出，
    // 用 stream2array_unpack(GEMM_TP=8) 正确还原为 (token, channel) 布局。
    stream2array_unpack<int8_t, AQ_T, 1, T, DEMUX_TP, C,  CP >(q_stream, DUT_XM_Q, "XM_Q", true);
    stream2array_unpack<int8_t, AS_T, 1, T, DEMUX_TP, CG, CPG>(s_stream, DUT_XM_S, "XM_S", true);
    assert(fc1_stream.empty());
    assert(q_stream  .empty());
    assert(s_stream  .empty());

    //* compare
    compare<int8_t>(REF_XM_Q, DUT_XM_Q, T*C,  "XM_Q");
    compare<int8_t>(REF_XM_S, DUT_XM_S, T*CG, "XM_S");
}

int main(){
    for(int l = 0; l < 1; ++l){
        test_layer(l);
        printf("layer %d passed\n", l);
    }
    return 0;
}
