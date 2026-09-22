#include "../src/common.h"
#include "../src/layernorm.h"
#include "../src/quantizer.h"

// ============================================================================
// ViT LayerNorm + Quantizer HLS 仿真测试 — 风格对齐 LLM_RMSNORM.cpp
// ============================================================================
// 每层有 2 个 LayerNorm (MHA 前 layernorm1 + MLP 前 layernorm2)
// ViT final/post LayerNorm 不在 PL 中实现；本独立模块只保留两个 pre-LayerNorm。
// 输入: CONDENSED_RESIDUAL_O.bin（VIT_X_T, int23, 2*T*C 个元素）
// 输出: CONDENSED_XLN_Q.bin（AQ_T, int8）+ CONDENSED_XLN_S.bin（AS_T, uint4）
// ============================================================================

// simulation hyperparameters
constexpr int L         = VIT_L;    // 12
constexpr int T_LOAD    = 5120;     // saved seq length (batch * patches)
constexpr int POS       = 0;        // start position (取第一张图的 1024 patches)

// model hyperparameters
constexpr int G         = VIT_G;    // 8
constexpr int C         = VIT_C;    // 768

// design hyperparameters
constexpr int T         = VIT_S;    // 1024
constexpr int TP        = 1;        // 每拍 1 token（除 MUX/RV_GEMM 外无需 TP>1）
constexpr int CP        = G;        // 8

// instantiate LAYERNORM and QUANTIZER
LAYERNORM <VIT_L, T, TP, C, CP   > layernorm_inst(VIT_LAYERNORM_LNW, VIT_LAYERNORM_LNB);
QUANTIZER<VIT_XLN_T, AQ_T, AS_T, 1, T, TP, C, CP, G> quantizer_inst;

// derived hyperparameters
constexpr int TT        = layernorm_inst.TT;    // 1024
constexpr int CT        = layernorm_inst.CT;    // 96
constexpr int CPG       = CP / G;               // 1（量化 scale 组宽）
constexpr int NUM_X     = 2 * T * C;            // 2*1024*768 = 1,572,864（两次 pass）

// top function
void top(
    // input scalars
    int l_begin,
    int l_close,
    // streams
    hls::stream<hls::vector<VIT_X_T, TP * CP> > &x_stream,
    hls::stream<hls::vector<AQ_T,    TP * CP> > &xlnq_stream,
    hls::stream<hls::vector<AS_T,    TP     > > &xlns_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=x_stream
    #pragma HLS interface axis port=xlnq_stream
    #pragma HLS interface axis port=xlns_stream

    #pragma HLS aggregate variable=x_stream    compact=bit
    #pragma HLS aggregate variable=xlnq_stream compact=bit
    #pragma HLS aggregate variable=xlns_stream compact=bit

    // 中间流：LAYERNORM 输出 → QUANTIZER 输入
    hls::stream<hls::vector<VIT_XLN_T, TP * CP> > xln_stream;

    for(int l=l_begin; l<l_close && l<L; ++l){
        for(int mha_or_mlp=0; mha_or_mlp<2; ++mha_or_mlp){
            #pragma HLS dataflow
            layernorm_inst.do_layernorm(l,     mha_or_mlp, x_stream,   xln_stream  );
            quantizer_inst.do_quant    (                   xln_stream, xlnq_stream, xlns_stream);
        }
    }
}

// declare ref data（全局避免栈溢出）
int64_t REF_X    [2 * VIT_S * VIT_C];   // condensed residual output（2 pass × T × C）
int8_t  REF_XLN_Q[2 * VIT_S * VIT_C];  // MHA + MLP 的量化输出 Q
int8_t  REF_XLN_S[2 * VIT_S * (VIT_C / VIT_G)]; // MHA + MLP 的量化 scale
// declare dut data
int8_t  DUT_XLN_Q[2 * VIT_S * VIT_C];
int8_t  DUT_XLN_S[2 * VIT_S * (VIT_C / VIT_G)];


void test_layer(int l){
    string file_path = VIT_BINARIES_PATH + to_string(l);
    string save_path = VIT_CONDENSE_PATH + to_string(l);

    {
        //* use condensed input from ViT_RESIDUAL（2 pass: pass0=MHA_X, pass1=MLP_X）
        auto CONDENSED_X = read_tensor<int64_t>(save_path + "/CONDENSED_RESIDUAL_O.bin");
        tensor2array<int64_t>(CONDENSED_X, REF_X, 1, 1, 1, 1, NUM_X, NUM_X);

        //* read reference outputs from binaries
        auto MHA_XLN_Q = read_tensor<int8_t>(file_path + "/MHA_XLN_Q.bin");
        auto MHA_XLN_S = read_tensor<int8_t>(file_path + "/MHA_XLN_S.bin");
        auto MLP_XLN_Q = read_tensor<int8_t>(file_path + "/MLP_XLN_Q.bin");
        auto MLP_XLN_S = read_tensor<int8_t>(file_path + "/MLP_XLN_S.bin");
        //             dtype,   tensor,     array,            H_LOAD, H, T_LOAD, T_START, T,  C_LOAD, C
        tensor2array<int8_t>(MHA_XLN_Q, REF_XLN_Q,          1, 1, T_LOAD, POS, T, C,  C );
        tensor2array<int8_t>(MHA_XLN_S, REF_XLN_S,          1, 1, T_LOAD, POS, T, CT, CT);
        tensor2array<int8_t>(MLP_XLN_Q, REF_XLN_Q + T*C,    1, 1, T_LOAD, POS, T, C,  C );
        tensor2array<int8_t>(MLP_XLN_S, REF_XLN_S + T*CT,   1, 1, T_LOAD, POS, T, CT, CT);
    }

    //* create streams
    hls::stream<hls::vector<VIT_X_T, TP * CP> > x_stream;
    hls::stream<hls::vector<AQ_T,    TP * CP> > xlnq_stream;
    hls::stream<hls::vector<AS_T,    TP     > > xlns_stream;

    //* write input stream（flat, TP*CP=8 wide，包含 2 pass 的数据）
    array2stream<int64_t, VIT_X_T, 1, 1, 1, 1, NUM_X, TP*CP>(REF_X, x_stream, "X", true);

    //* call top function
    top(l, l+1, x_stream, xlnq_stream, xlns_stream);

    //* save condensed output
    save_condensed_tensor<int8_t, AQ_T, 2*T*C,  TP*CP>(save_path + "/CONDENSED_XLN_Q.bin", xlnq_stream);
    save_condensed_tensor<int8_t, AS_T, 2*T*CT, TP   >(save_path + "/CONDENSED_XLN_S.bin", xlns_stream);

    //* read output for comparison
    stream2array<int8_t, AQ_T, 2, T, TP, C,  CP>(xlnq_stream, DUT_XLN_Q, "XLN_Q", true);
    stream2array<int8_t, AS_T, 2, T, TP, CT, 1 >(xlns_stream, DUT_XLN_S, "XLN_S", true);
    assert(x_stream   .empty());
    assert(xlnq_stream.empty());
    assert(xlns_stream.empty());

    //* compare
    compare<int8_t>(REF_XLN_Q, DUT_XLN_Q, 2*T*C,  "XLN_Q");
    compare<int8_t>(REF_XLN_S, DUT_XLN_S, 2*T*CT, "XLN_S");
}

int main(){
    for(int l=0; l<1; ++l){
        test_layer(l);
        printf("Test %d passed\n", l);
    }
    return 0;
}
