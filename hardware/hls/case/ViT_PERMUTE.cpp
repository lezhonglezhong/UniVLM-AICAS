#include "../src/common.h"
#include "../src/vit_tensor_core.h"
#include "../src/vit_accumulator.h"
#include "../src/utils.h"

// ============================================================================
// Vision Encoder (ViT) GEMM_PERMUTE - 有 Bias（但 bias 加法在别处处理）
// ============================================================================
// SigLIP ViT: Q/K/V/Out/FC1/FC2 均有 bias（由 ViT_RESIDUAL 等模块处理）
// 注意：Text Decoder 版本在 LLM_PERMUTE.cpp 中（无 bias）

// 设计的consideration：
// 1. 权重的全广播：为了实现单周期消耗掉读入权重的所有数据依赖，必须将该权重广播到所有token，T被全展开，T==TP
// 2. unpack：gemm产生的输出数据位宽非常高，accumulator每隔很多个周期才会产生一个结果。unpack就是"完全解开T维度"的输出，而下一级需要再pack回来。
// 3. 多层设计与多层仿真：模块仅输入l_begin和l_close，自行判断模式
// 4. ViT MLP 为 fc1(768→3072)+GELU+fc2(3072→768)，无门控（不同于 LLM SwiGLU）
// 5. 去掉了 CLS / embedding / Connector 路径，仅保留 encoder

// simulation hyperparameters
constexpr int L             = VIT_L;       // 12
constexpr int T_LOAD        = 5120;        // 5 images * 1024 patches  B_LOAD=5
constexpr int POS           = 0;           // start position

// model hyperparameters
constexpr int H             = VIT_H;       // 12
constexpr int G             = VIT_G;       // 8
constexpr int C             = VIT_C;       // 768
constexpr int HC            = VIT_HC;      // 64
constexpr int CM            = VIT_MLP_DIM; // 3072

// design hyperparameters
constexpr int T             = 1024;  // 1024 (total tokens)
constexpr int TP            = 8;      // hardware token parallelism (replaces old T=8)
constexpr int TT            = T / TP; // token tiles
constexpr int CIP           = G;
constexpr int COP           = G;
// Note: DW_AQ=8, DW_WQ=5, DW_AS=4, DW_WS=4 are the same for ViT and LLAMA,
// so we reuse the existing DW_AQ/DW_WQ/DW_AS/DW_WS from common.h.

// TRUNC_BASE = min of all ViT encoder path truncations = 9
// = min(QK=13, V=14, O=19, FC1=10, FC2=17, EMB=9)
constexpr int TRUNC_BASE    = VIT_TRUNC_BASE;

// ViT GEMM accumulator bit width (different from LLAMA's DW_GEMM_ACC=64)
// VIT: max(VIT_DW_QKV=34, VIT_DW_O=36, VIT_DW_XUG=38, VIT_DW_XD=40) = 40
// DW_GEMM = DW_AQ + DW_WQ + log2ce(G) = 8+5+3=16, same as LLAMA, reuse DW_GEMM from common.h.

// derived hyperparameters
constexpr int CT            = C  / CIP;   // 96
constexpr int CMT           = CM / CIP;   // 384
constexpr int HCT           = HC / CIP;   // 8

// data types (DW_AQ=8, DW_WQ=5, DW_AS/WS=4, DW_GEMM=16 same as LLAMA, reused from common.h)
typedef ap_int  <DW_AQ           > aq_t;
typedef ap_uint <DW_AS           > as_t;
typedef ap_int  <DW_WQ           > wq_t;
typedef ap_uint <DW_WS           > ws_t;
typedef ap_int  <VIT_DW_GEMM     > gemm_t;
typedef ap_int  <VIT_DW_GEMM_ACC     > acc_t;   // 40 (ViT-specific, differs from LLAMA's 64)
typedef ap_int  <VIT_DW_GEMM_TRUNC   > of_t;    // 31 (ViT-specific)

// derived simulation hyperparameters
// ViT: QKVO + FC1 + FC2
// QKVO:  T*C  -> T*C   (fc1, fc2 of attn are both C→C)
// FC1:   T*C  -> T*CM  (single, no gate unlike LLM)
// FC2:   T*CM -> T*C
// 注意：T=1024 时 T*C*CM 等乘积超过 32-bit int，须用 long long 再收窄，否则 CSIM 报 overflow
constexpr long long T_QKVO_LL =
    (static_cast<long long>(T) * C * C) / (TP * CIP * COP);
constexpr long long T_FC1_LL =
    (static_cast<long long>(T) * C * CM) / (TP * CIP * COP);
constexpr long long T_FC2_LL =
    (static_cast<long long>(T) * CM * C) / (TP * CIP * COP);
const int T_QKVO        = static_cast<int>(T_QKVO_LL);
const int T_FC1         = static_cast<int>(T_FC1_LL);
const int T_FC2         = static_cast<int>(T_FC2_LL);
const int T_DECODER     = T_QKVO * 4 + T_FC1 + T_FC2;  // whole-layer iterations

// accumulator: CYCS = input channel tripcount; REPEAT = output channel groups
const int ACC_CYCS1     = C      / CIP;  // 96  (for QKVO and FC1 input)
const int ACC_CYCS2     = CM     / CIP;  // 384 (for FC2 input)
const int N_QKVO        = C  / COP;      // 96
const int N_FC1         = CM / COP;      // 384
const int N_FC2         = C  / COP;      // 96
const int ACC_REPEAT1   = TT * (N_QKVO*4 + N_FC1);   //理解成块的数量
const int ACC_REPEAT2   = TT * N_FC2;

VIT_TENSOR_CORE <aq_t, wq_t, gemm_t, TP, CIP, COP, T_DECODER> tensor_core_inst;
VIT_ACCUMULATOR <gemm_t, as_t, ws_t, acc_t, of_t, TRUNC_BASE, TP, COP, ACC_CYCS1, ACC_CYCS2, ACC_REPEAT1, ACC_REPEAT2> accumulator_inst;

void top_helper(
    hls::stream<hls::vector<aq_t,    TP *CIP> >& i_stream,
    hls::stream<hls::vector<as_t,    TP     > >& s_stream,
    hls::stream<hls::vector<wq_t,    COP*CIP> >& w_stream,
    hls::stream<hls::vector<ws_t,    COP    > >& s1_stream,
    hls::stream<hls::vector<ws_t,    COP    > >& s2_stream,
    hls::stream<hls::vector<of_t,    COP    > >& o_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=i_stream
    #pragma HLS interface axis port=w_stream
    #pragma HLS interface axis port=s_stream
    #pragma HLS interface axis port=s1_stream
    #pragma HLS interface axis port=s2_stream
    #pragma HLS interface axis port=o_stream
    #pragma HLS aggregate variable=i_stream  compact=bit
    #pragma HLS aggregate variable=w_stream  compact=bit
    #pragma HLS aggregate variable=s_stream  compact=bit
    #pragma HLS aggregate variable=s1_stream compact=bit
    #pragma HLS aggregate variable=s2_stream compact=bit
    #pragma HLS aggregate variable=o_stream  compact=bit

    hls::stream<hls::vector<gemm_t, TP*COP> > gemm_stream ("gemm_stream");
    #pragma HLS dataflow
    tensor_core_inst.do_tensor_core(i_stream, w_stream, gemm_stream);
    accumulator_inst.do_accumulator(gemm_stream, s_stream, s1_stream, s2_stream, o_stream);
}

// top function
void top(
    int l_begin,
    int l_close,
    hls::stream<hls::vector<aq_t,    TP *CIP> >& i_stream,
    hls::stream<hls::vector<as_t,    TP     > >& s_stream,
    hls::stream<hls::vector<wq_t,    COP*CIP> >& w_stream,
    hls::stream<hls::vector<ws_t,    COP    > >& s1_stream,
    hls::stream<hls::vector<ws_t,    COP    > >& s2_stream,
    hls::stream<hls::vector<of_t,    COP    > >& o_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=i_stream
    #pragma HLS interface axis port=w_stream
    #pragma HLS interface axis port=s_stream
    #pragma HLS interface axis port=s1_stream
    #pragma HLS interface axis port=s2_stream
    #pragma HLS interface axis port=o_stream
    #pragma HLS aggregate variable=i_stream  compact=bit
    #pragma HLS aggregate variable=w_stream  compact=bit
    #pragma HLS aggregate variable=s_stream  compact=bit
    #pragma HLS aggregate variable=s1_stream compact=bit
    #pragma HLS aggregate variable=s2_stream compact=bit
    #pragma HLS aggregate variable=o_stream  compact=bit

    for(int l=l_begin; l<l_close; ++l){
        top_helper(i_stream, s_stream, w_stream, s1_stream, s2_stream, o_stream);
    }
}


// ============================================================================
// Reference data arrays
// ============================================================================

// ViT encoder layer inputs
int8_t  REF_MHA_XLN_Q   [L][T *C    ];
int8_t  REF_MHA_XLN_S   [L][T *CT   ];
int8_t  REF_MHA_A_Q     [L][T *C    ];  // attention output (input to O proj)
int8_t  REF_MHA_A_S     [L][T *CT   ];
int8_t  REF_MLP_XLN_Q   [L][T *C    ];
int8_t  REF_MLP_XLN_S   [L][T *CT   ];
int8_t  REF_MLP_XM_Q    [L][T *CM   ];  // GELU output (input to fc2)
int8_t  REF_MLP_XM_S    [L][T *CMT  ];

// ViT encoder weights (per layer, declared globally for reuse)
int8_t  REF_WQ_Q           [H * HC * C  ];
int8_t  REF_WQ_S1          [H * HC * CT ];
int8_t  REF_WQ_S2          [H * HC * CT ];
int8_t  REF_WK_Q           [H * HC * C  ];
int8_t  REF_WK_S1          [H * HC * CT ];
int8_t  REF_WK_S2          [H * HC * CT ];
int8_t  REF_WV_Q           [H * HC * C  ];
int8_t  REF_WV_S1          [H * HC * CT ];
int8_t  REF_WV_S2          [H * HC * CT ];
int8_t  REF_WO_Q           [C  * C      ];
int8_t  REF_WO_S1          [C  * CT     ];
int8_t  REF_WO_S2          [C  * CT     ];
int8_t  REF_WFC1_Q         [CM * C      ];  // fc1: [3072, 768]
int8_t  REF_WFC1_S1        [CM * CT     ];
int8_t  REF_WFC1_S2        [CM * CT     ];
int8_t  REF_WFC2_Q         [C  * CM     ];  // fc2: [768, 3072]
int8_t  REF_WFC2_S1        [C  * CMT    ];
int8_t  REF_WFC2_S2        [C  * CMT    ];

// ViT encoder results
int64_t REF_MHA_Q       [L][T * H * HC];
int64_t REF_MHA_K       [L][T * H * HC];
int64_t REF_MHA_V       [L][T * H * HC];
int64_t REF_MHA_O       [L][T * C     ];
int64_t REF_MHA_BQ      [L][C];
int64_t REF_MHA_BK      [L][C];
int64_t REF_MHA_BV      [L][C];
int64_t REF_MHA_BO      [L][C];
int64_t REF_MLP_B1      [L][CM];
int64_t REF_MLP_B2      [L][C];
int64_t REF_MLP_FC1     [L][T * CM    ];  // fc1 output (single, no gate)
int64_t REF_MLP_FC2     [L][T * C     ];  // fc2 output

// ViT encoder permuted refs
int64_t REF_MHA_PERM_QKV[L][H  *3*T*HC  ];
int64_t REF_MHA_PERM_O  [L][H    *T*HC  ];
int64_t REF_MLP_PERM_FC1[L][    T*CM ];  // single projection (no gate)
int64_t REF_MLP_PERM_FC2[L][      T*C   ];

// ViT encoder permuted DUT output
int64_t DUT_MHA_PERM_QKV   [H  *3*T*HC  ];
int64_t DUT_MHA_PERM_O     [H    *T*HC  ];
int64_t DUT_MLP_PERM_FC1   [T*CM ];
int64_t DUT_MLP_PERM_FC2   [      T*C   ];

// ============================================================================
// prepare_encoder_data — encoder layer GEMM input/weight stream setup
// ============================================================================
void prepare_encoder_data(
    int l,
    hls::stream<hls::vector<aq_t,    TP *CIP> >& i_stream,
    hls::stream<hls::vector<as_t,    TP     > >& s_stream,
    hls::stream<hls::vector<wq_t,    COP*CIP> >& w_stream,
    hls::stream<hls::vector<ws_t,    COP    > >& s1_stream,
    hls::stream<hls::vector<ws_t,    COP    > >& s2_stream
){
    const string binaries_path = VIT_BINARIES_PATH + to_string(l);
    const string condense_path = VIT_CONDENSE_PATH + to_string(l);

    auto MHA_XLN_Q  = read_tensor<int8_t> (binaries_path + "/MHA_XLN_Q.bin"   );
    auto MHA_XLN_S  = read_tensor<int8_t> (binaries_path + "/MHA_XLN_S.bin"   );
    auto MHA_A_Q    = read_tensor<int8_t> (binaries_path + "/MHA_A_Q.bin"     );
    auto MHA_A_S    = read_tensor<int8_t> (binaries_path + "/MHA_A_S.bin"     );
    auto MLP_XLN_Q  = read_tensor<int8_t> (binaries_path + "/MLP_XLN_Q.bin"   );
    auto MLP_XLN_S  = read_tensor<int8_t> (binaries_path + "/MLP_XLN_S.bin"   );
    auto MLP_XM_Q   = read_tensor<int8_t> (binaries_path + "/MLP_XM_Q.bin"    );
    auto MLP_XM_S   = read_tensor<int8_t> (binaries_path + "/MLP_XM_S.bin"    );
    // weights
    auto MHA_WQ_Q   = read_tensor<int8_t> (binaries_path + "/MHA_WQ_Q.bin"    );
    auto MHA_WQ_S1  = read_tensor<int8_t> (binaries_path + "/MHA_WQ_S1.bin"   );
    auto MHA_WQ_S2  = read_tensor<int8_t> (binaries_path + "/MHA_WQ_S2.bin"   );
    auto MHA_WK_Q   = read_tensor<int8_t> (binaries_path + "/MHA_WK_Q.bin"    );
    auto MHA_WK_S1  = read_tensor<int8_t> (binaries_path + "/MHA_WK_S1.bin"   );
    auto MHA_WK_S2  = read_tensor<int8_t> (binaries_path + "/MHA_WK_S2.bin"   );
    auto MHA_WV_Q   = read_tensor<int8_t> (binaries_path + "/MHA_WV_Q.bin"    );
    auto MHA_WV_S1  = read_tensor<int8_t> (binaries_path + "/MHA_WV_S1.bin"   );
    auto MHA_WV_S2  = read_tensor<int8_t> (binaries_path + "/MHA_WV_S2.bin"   );
    auto MHA_WO_Q   = read_tensor<int8_t> (binaries_path + "/MHA_WO_Q.bin"    );
    auto MHA_WO_S1  = read_tensor<int8_t> (binaries_path + "/MHA_WO_S1.bin"   );
    auto MHA_WO_S2  = read_tensor<int8_t> (binaries_path + "/MHA_WO_S2.bin"   );
    auto MLP_WFC1_Q = read_tensor<int8_t> (binaries_path + "/MLP_WFC1_Q.bin"  );
    auto MLP_WFC1_S1= read_tensor<int8_t> (binaries_path + "/MLP_WFC1_S1.bin" );
    auto MLP_WFC1_S2= read_tensor<int8_t> (binaries_path + "/MLP_WFC1_S2.bin" );
    auto MLP_WFC2_Q = read_tensor<int8_t> (binaries_path + "/MLP_WFC2_Q.bin"  );
    auto MLP_WFC2_S1= read_tensor<int8_t> (binaries_path + "/MLP_WFC2_S1.bin" );
    auto MLP_WFC2_S2= read_tensor<int8_t> (binaries_path + "/MLP_WFC2_S2.bin" );
    // results
    auto MHA_Q      = read_tensor<int64_t>(binaries_path + "/MHA_Q.bin"       );
    auto MHA_K      = read_tensor<int64_t>(binaries_path + "/MHA_K.bin"       );
    auto MHA_V      = read_tensor<int64_t>(binaries_path + "/MHA_V.bin"       );
    auto MHA_O      = read_tensor<int64_t>(binaries_path + "/MHA_O.bin"       );
    auto MHA_BQ     = read_tensor<int64_t>(binaries_path + "/MHA_BQ.bin"      );
    auto MHA_BK     = read_tensor<int64_t>(binaries_path + "/MHA_BK.bin"      );
    auto MHA_BV     = read_tensor<int64_t>(binaries_path + "/MHA_BV.bin"      );
    auto MHA_BO     = read_tensor<int64_t>(binaries_path + "/MHA_BO.bin"      );
    auto MLP_B1     = read_tensor<int64_t>(binaries_path + "/MLP_B1.bin"      );
    auto MLP_B2     = read_tensor<int64_t>(binaries_path + "/MLP_B2.bin"      );
    auto MLP_XFC1   = read_tensor<int64_t>(binaries_path + "/MLP_XFC1.bin"    );
    auto MLP_XFC2   = read_tensor<int64_t>(binaries_path + "/MLP_XFC2.bin"    );

    // fill input refs
    tensor2array<int8_t> (MHA_XLN_Q,  REF_MHA_XLN_Q[l], 1, 1, T_LOAD, POS, T, C,   C  );
    tensor2array<int8_t> (MHA_XLN_S,  REF_MHA_XLN_S[l], 1, 1, T_LOAD, POS, T, CT,  CT );
    tensor2array<int8_t> (MHA_A_Q,    REF_MHA_A_Q  [l], 1, 1, T_LOAD, POS, T, C,   C  );
    tensor2array<int8_t> (MHA_A_S,    REF_MHA_A_S  [l], 1, 1, T_LOAD, POS, T, CT,  CT );
    tensor2array<int8_t> (MLP_XLN_Q,  REF_MLP_XLN_Q[l], 1, 1, T_LOAD, POS, T, C,   C  );
    tensor2array<int8_t> (MLP_XLN_S,  REF_MLP_XLN_S[l], 1, 1, T_LOAD, POS, T, CT,  CT );
    tensor2array<int8_t> (MLP_XM_Q,   REF_MLP_XM_Q [l], 1, 1, T_LOAD, POS, T, CM,  CM );
    tensor2array<int8_t> (MLP_XM_S,   REF_MLP_XM_S [l], 1, 1, T_LOAD, POS, T, CMT, CMT);
    // fill weight refs
    tensor2array<int8_t> (MHA_WQ_Q,   REF_WQ_Q,  1, 1, H*HC, H*HC, C,   C  );
    tensor2array<int8_t> (MHA_WQ_S1,  REF_WQ_S1, 1, 1, H*HC, H*HC, CT,  CT );
    tensor2array<int8_t> (MHA_WQ_S2,  REF_WQ_S2, 1, 1, H*HC, H*HC, CT,  CT );
    tensor2array<int8_t> (MHA_WK_Q,   REF_WK_Q,  1, 1, H*HC, H*HC, C,   C  );
    tensor2array<int8_t> (MHA_WK_S1,  REF_WK_S1, 1, 1, H*HC, H*HC, CT,  CT );
    tensor2array<int8_t> (MHA_WK_S2,  REF_WK_S2, 1, 1, H*HC, H*HC, CT,  CT );
    tensor2array<int8_t> (MHA_WV_Q,   REF_WV_Q,  1, 1, H*HC, H*HC, C,   C  );
    tensor2array<int8_t> (MHA_WV_S1,  REF_WV_S1, 1, 1, H*HC, H*HC, CT,  CT );
    tensor2array<int8_t> (MHA_WV_S2,  REF_WV_S2, 1, 1, H*HC, H*HC, CT,  CT );
    tensor2array<int8_t> (MHA_WO_Q,   REF_WO_Q,  1, 1, C,    C,    C,   C  );
    tensor2array<int8_t> (MHA_WO_S1,  REF_WO_S1, 1, 1, C,    C,    CT,  CT );
    tensor2array<int8_t> (MHA_WO_S2,  REF_WO_S2, 1, 1, C,    C,    CT,  CT );
    tensor2array<int8_t> (MLP_WFC1_Q, REF_WFC1_Q,  1, 1, CM, CM, C,   C  );
    tensor2array<int8_t> (MLP_WFC1_S1,REF_WFC1_S1, 1, 1, CM, CM, CT,  CT );
    tensor2array<int8_t> (MLP_WFC1_S2,REF_WFC1_S2, 1, 1, CM, CM, CT,  CT );
    tensor2array<int8_t> (MLP_WFC2_Q, REF_WFC2_Q,  1, 1, C,  C,  CM,  CM );
    tensor2array<int8_t> (MLP_WFC2_S1,REF_WFC2_S1, 1, 1, C,  C,  CMT, CMT);
    tensor2array<int8_t> (MLP_WFC2_S2,REF_WFC2_S2, 1, 1, C,  C,  CMT, CMT);
    // fill result refs
    tensor2array<int64_t>(MHA_Q,   REF_MHA_Q  [l], 1, 1, T_LOAD, POS, T, H*HC, H*HC);
    tensor2array<int64_t>(MHA_K,   REF_MHA_K  [l], 1, 1, T_LOAD, POS, T, H*HC, H*HC);
    tensor2array<int64_t>(MHA_V,   REF_MHA_V  [l], 1, 1, T_LOAD, POS, T, H*HC, H*HC);
    tensor2array<int64_t>(MHA_O,   REF_MHA_O  [l], 1, 1, T_LOAD, POS, T, C,    C   );
    tensor2array<int64_t>(MHA_BQ,  REF_MHA_BQ [l], 1, 1, 1,           1, C,    C   );
    tensor2array<int64_t>(MHA_BK,  REF_MHA_BK [l], 1, 1, 1,           1, C,    C   );
    tensor2array<int64_t>(MHA_BV,  REF_MHA_BV [l], 1, 1, 1,           1, C,    C   );
    tensor2array<int64_t>(MHA_BO,  REF_MHA_BO [l], 1, 1, 1,           1, C,    C   );
    tensor2array<int64_t>(MLP_B1,  REF_MLP_B1 [l], 1, 1, 1,           1, CM,   CM  );
    tensor2array<int64_t>(MLP_B2,  REF_MLP_B2 [l], 1, 1, 1,           1, C,    C   );
    tensor2array<int64_t>(MLP_XFC1,REF_MLP_FC1[l], 1, 1, T_LOAD, POS, T, CM,   CM  );
    tensor2array<int64_t>(MLP_XFC2,REF_MLP_FC2[l], 1, 1, T_LOAD, POS, T, C,    C   );

    //* HANDCRAFTED PERMUTATION
    //* for QKV, divide output channel in the granularity of head
    for(int h=0; h<H; ++h){
        for(int qkv=0; qkv<3; ++qkv){
            for(int t=0; t<T; ++t){
                for(int hc=0; hc<HC; ++hc){
                    REF_MHA_PERM_QKV[l][h*3*T*HC + qkv*T*HC + t*HC + hc] =
                        qkv == 0 ? REF_MHA_Q[l][t*C + h*HC + hc] :
                        qkv == 1 ? REF_MHA_K[l][t*C + h*HC + hc] :
                                   REF_MHA_V[l][t*C + h*HC + hc];
                }
            }
        }
    }
    //* for O, linear order (same style as FC1/FC2: no explicit head split in stream order)
    for(int t=0; t<T; ++t){
        for(int c=0; c<C; ++c){
            REF_MHA_PERM_O[l][t*C + c] = REF_MHA_O[l][t*C + c];
        }
    }

    for(int t=0; t<T; ++t){
        for(int c=0; c<CM; ++c){
            REF_MLP_PERM_FC1[l][t*CM + c] = REF_MLP_FC1[l][t*CM + c];
        }
    }

    //* for FC2, normal (no head split)
    for(int t=0; t<T; ++t){
        for(int c=0; c<C; ++c){
            REF_MLP_PERM_FC2[l][t*C + c] = REF_MLP_FC2[l][t*C + c];
        }
    }

    //* declare simulated streams
    hls::stream<hls::vector<aq_t,    TP *CIP> > sim_i_stream ("sim_i_stream");
    hls::stream<hls::vector<as_t,    TP     > > sim_s_stream ("sim_s_stream");
    hls::stream<hls::vector<wq_t,    COP*CIP> > sim_w_stream ("sim_w_stream");
    hls::stream<hls::vector<ws_t,    COP    > > sim_s1_stream("sim_s1_stream");
    hls::stream<hls::vector<ws_t,    COP    > > sim_s2_stream("sim_s2_stream");

    //* write input data to stream
    //* enforce block-major order expected by unpack:
    //* QKV blocks: h -> qkv -> tt -> hct(out) -> ct(in)
    for(int h=0; h<H; ++h){
        for(int qkv=0; qkv<3; ++qkv){
            for(int tt=0; tt<TT; ++tt){
                for(int hct=0; hct<HCT; ++hct){
                    (void)hct;
                    for(int ct=0; ct<CT; ++ct){
                        hls::vector<aq_t, TP*CIP> i_vec;
                        hls::vector<as_t, TP> s_vec;
                        for(int tp=0; tp<TP; ++tp){
                            for(int cip=0; cip<CIP; ++cip){
                                i_vec[tp*CIP + cip] = REF_MHA_XLN_Q[l][(tt*TP + tp)*C + ct*CIP + cip];
                            }
                            s_vec[tp] = REF_MHA_XLN_S[l][(tt*TP + tp)*CT + ct];
                        }
                        sim_i_stream.write(i_vec);
                        sim_s_stream.write(s_vec);
                    }
                }
            }
        }
    }
    //* O blocks: tt -> ct_out(out) -> ct(in)
    for(int tt=0; tt<TT; ++tt){
        for(int ct_out=0; ct_out<CT; ++ct_out){
            (void)ct_out;
            for(int ct=0; ct<CT; ++ct){
                hls::vector<aq_t, TP*CIP> i_vec;
                hls::vector<as_t, TP> s_vec;
                for(int tp=0; tp<TP; ++tp){
                    for(int cip=0; cip<CIP; ++cip){
                        i_vec[tp*CIP + cip] = REF_MHA_A_Q[l][(tt*TP + tp)*C + ct*CIP + cip];
                    }
                    s_vec[tp] = REF_MHA_A_S[l][(tt*TP + tp)*CT + ct];
                }
                sim_i_stream.write(i_vec);
                sim_s_stream.write(s_vec);
            }
        }
    }
    //* FC1 blocks: tt -> cmt(out) -> ct(in)
    for(int tt=0; tt<TT; ++tt){
        for(int cmt=0; cmt<CMT; ++cmt){
            for(int ct=0; ct<CT; ++ct){
                hls::vector<aq_t, TP*CIP> i_vec;
                hls::vector<as_t, TP> s_vec;
                for(int tp=0; tp<TP; ++tp){
                    for(int cip=0; cip<CIP; ++cip){
                        i_vec[tp*CIP + cip] = REF_MLP_XLN_Q[l][(tt*TP + tp)*C + ct*CIP + cip];
                    }
                    s_vec[tp] = REF_MLP_XLN_S[l][(tt*TP + tp)*CT + ct];
                }
                sim_i_stream.write(i_vec);
                sim_s_stream.write(s_vec);
            }
        }
    }
    //* FC2 blocks: tt -> ct_out(out) -> cmt(in)
    for(int tt=0; tt<TT; ++tt){
        for(int ct_out=0; ct_out<CT; ++ct_out){
            (void)ct_out;
            for(int cmt=0; cmt<CMT; ++cmt){
                hls::vector<aq_t, TP*CIP> i_vec;
                hls::vector<as_t, TP> s_vec;
                for(int tp=0; tp<TP; ++tp){
                    for(int cip=0; cip<CIP; ++cip){
                        i_vec[tp*CIP + cip] = REF_MLP_XM_Q[l][(tt*TP + tp)*CM + cmt*CIP + cip];
                    }
                    s_vec[tp] = REF_MLP_XM_S[l][(tt*TP + tp)*CMT + cmt];
                }
                sim_i_stream.write(i_vec);
                sim_s_stream.write(s_vec);
            }
        }
    }

    //* QKV weights/shift: make head-major order, then replicate across token tiles
    for(int h=0; h<H; ++h){
        for(int qkv=0; qkv<3; ++qkv){
            int8_t* wq_ptr   = (qkv == 0) ? (REF_WQ_Q  + h*HC*C)  : (qkv == 1) ? (REF_WK_Q  + h*HC*C)  : (REF_WV_Q  + h*HC*C);
            int8_t* ws1_ptr  = (qkv == 0) ? (REF_WQ_S1 + h*HC*CT) : (qkv == 1) ? (REF_WK_S1 + h*HC*CT) : (REF_WV_S1 + h*HC*CT);
            int8_t* ws2_ptr  = (qkv == 0) ? (REF_WQ_S2 + h*HC*CT) : (qkv == 1) ? (REF_WK_S2 + h*HC*CT) : (REF_WV_S2 + h*HC*CT);
            for(int tt=0; tt<TT; ++tt){
                array2stream<int8_t, wq_t, 1, 1, HC, COP, C,  CIP>(wq_ptr,  sim_w_stream);
                array2stream<int8_t, ws_t, 1, 1, HC, COP, CT, 1  >(ws1_ptr, sim_s1_stream);
                array2stream<int8_t, ws_t, 1, 1, HC, COP, CT, 1  >(ws2_ptr, sim_s2_stream);
            }
        }
    }

    //* O weights/shift: tt -> ct_out(out) -> ct(in)
    for(int tt=0; tt<TT; ++tt){
        for(int ct_out=0; ct_out<CT; ++ct_out){
            array2stream<int8_t, wq_t, 1, 1, COP, COP, C,  CIP>(REF_WO_Q  + ct_out*COP*C,  sim_w_stream);
            array2stream<int8_t, ws_t, 1, 1, COP, COP, CT, 1  >(REF_WO_S1 + ct_out*COP*CT, sim_s1_stream);
            array2stream<int8_t, ws_t, 1, 1, COP, COP, CT, 1  >(REF_WO_S2 + ct_out*COP*CT, sim_s2_stream);
        }
    }
    //* FC1 weights: tt -> cmt(out) -> ct(in)
    for(int tt=0; tt<TT; ++tt){
        for(int cmt=0; cmt<CMT; ++cmt){
            array2stream<int8_t, wq_t, 1, 1, COP, COP, C,  CIP>(REF_WFC1_Q  + cmt*COP*C,  sim_w_stream);
            array2stream<int8_t, ws_t, 1, 1, COP, COP, CT, 1  >(REF_WFC1_S1 + cmt*COP*CT, sim_s1_stream);
            array2stream<int8_t, ws_t, 1, 1, COP, COP, CT, 1  >(REF_WFC1_S2 + cmt*COP*CT, sim_s2_stream);
        }
    }
    //* FC2 weights: tt -> ct_out(out) -> cmt(in)
    for(int tt=0; tt<TT; ++tt){
        for(int ct_out=0; ct_out<CT; ++ct_out){
            array2stream<int8_t, wq_t, 1, 1, COP, COP, CM,  CIP>(REF_WFC2_Q  + ct_out*COP*CM,  sim_w_stream);
            array2stream<int8_t, ws_t, 1, 1, COP, COP, CMT, 1  >(REF_WFC2_S1 + ct_out*COP*CMT, sim_s1_stream);
            array2stream<int8_t, ws_t, 1, 1, COP, COP, CMT, 1  >(REF_WFC2_S2 + ct_out*COP*CMT, sim_s2_stream);
        }
    }

    //* save condensed input and weight
    const int NUM_X  = (3*CT*T*C + CT*T*C + CMT*T*C + CT*T*CM);
    const int NUM_W  = TT * (3*C*C    + C*C    + CM*C    + C*CM   );
    const int NUM_XS = NUM_X / CIP;
    const int NUM_WS = NUM_W / CIP;
    save_condensed_tensor<int8_t, aq_t, NUM_X,  TP *CIP>(condense_path + "/CONDENSED_GEMM_X_Q.bin" , sim_i_stream );
    save_condensed_tensor<int8_t, as_t, NUM_XS, TP     >(condense_path + "/CONDENSED_GEMM_X_S.bin" , sim_s_stream );
    save_condensed_tensor<int8_t, wq_t, NUM_W,  COP*CIP>(condense_path + "/CONDENSED_GEMM_W_Q.bin" , sim_w_stream );
    save_condensed_tensor<int8_t, ws_t, NUM_WS, COP    >(condense_path + "/CONDENSED_GEMM_W_S1.bin", sim_s1_stream);
    save_condensed_tensor<int8_t, ws_t, NUM_WS, COP    >(condense_path + "/CONDENSED_GEMM_W_S2.bin", sim_s2_stream);

    //* move simulated streams to real streams
    stream2stream<aq_t, TP *CIP>(sim_i_stream,  i_stream );
    stream2stream<as_t, TP     >(sim_s_stream,  s_stream );
    stream2stream<wq_t, COP*CIP>(sim_w_stream,  w_stream );
    stream2stream<ws_t, COP    >(sim_s1_stream, s1_stream);
    stream2stream<ws_t, COP    >(sim_s2_stream, s2_stream);
}


// ============================================================================
// compare_encoder_data
// ============================================================================
void compare_encoder_data(
    int l,
    hls::stream<hls::vector<of_t, COP> >& o_stream
){
    const string condense_path = VIT_CONDENSE_PATH + to_string(l);
    //* save condensed output
    const int NUM_Y = (H*3*T*HC + H*T*HC + T*CM + T*C);  // QKV + O + FC1 + FC2
    hls::stream<hls::vector<of_t, COP> > sim_o_stream("sim_o_stream");
    stream2stream        <         of_t, NUM_Y, COP>(o_stream, sim_o_stream);
    save_condensed_tensor<int64_t, of_t, NUM_Y, COP>(condense_path + "/CONDENSED_GEMM_Y.bin", sim_o_stream);
    //* unpack output (TP=8 tokens per tile, T/TP=128 tiles)
    stream2array_unpack<int64_t, of_t, H*3, T, TP, HC,  COP>(sim_o_stream, DUT_MHA_PERM_QKV, "Output QKV", true);
    stream2array_unpack<int64_t, of_t, 1,   T, TP, C,   COP>(sim_o_stream, DUT_MHA_PERM_O,   "Output O",   true);  //LLM因为没有tt，h和hct可以随时乘在一起，所以unpack成分头的，也只有这里有验证的作用
    stream2array_unpack<int64_t, of_t, 1,   T, TP, CM,  COP>(sim_o_stream, DUT_MLP_PERM_FC1, "Output FC1", true);
    stream2array_unpack<int64_t, of_t, 1,   T, TP, C,   COP>(sim_o_stream, DUT_MLP_PERM_FC2, "Output FC2", true);
    assert(sim_o_stream.size() == 0);
    //* extra truncation per path
    for(int h=0; h<H; ++h){
        for(int qkv=0; qkv<3; ++qkv){
            int extra_trunc = (qkv==0 || qkv==1) ? (VIT_MHA_TRUNC_QK - TRUNC_BASE)
                                                  : (VIT_MHA_TRUNC_V  - TRUNC_BASE);
            truncate<int64_t>(DUT_MHA_PERM_QKV + h*3*T*HC + qkv*T*HC, T*HC, extra_trunc);
        }
    }
    for(int h=0; h<H; ++h){
        for(int qkv=0; qkv<3; ++qkv){
            for(int hc=0; hc<HC; ++hc){
                for(int t=0; t<T; ++t){
                    int idx = h*3*T*HC + qkv*T*HC + t*HC + hc;
                    int c_idx = h*HC + hc;
                    int64_t bias = (qkv == 0) ? REF_MHA_BQ[l][c_idx]
                                              : (qkv == 1) ? REF_MHA_BK[l][c_idx]
                                                           : REF_MHA_BV[l][c_idx];
                    DUT_MHA_PERM_QKV[idx] += bias;
                }
            }
        }
    }
    truncate<int64_t>(DUT_MHA_PERM_O,   T*C,  (VIT_MHA_TRUNC_O   - TRUNC_BASE));
    for(int t=0; t<T; ++t){
        for(int c=0; c<C; ++c){
            DUT_MHA_PERM_O[t*C + c] += REF_MHA_BO[l][c];
        }
    }
    truncate<int64_t>(DUT_MLP_PERM_FC1, T*CM,(VIT_MLP_TRUNC_FC1 - TRUNC_BASE));
    for(int t=0; t<T; ++t){
        for(int c=0; c<CM; ++c){
            DUT_MLP_PERM_FC1[t*CM + c] += REF_MLP_B1[l][c];
        }
    }
    truncate<int64_t>(DUT_MLP_PERM_FC2, T*C,      (VIT_MLP_TRUNC_FC2 - TRUNC_BASE));
    for(int t=0; t<T; ++t){
        for(int c=0; c<C; ++c){
            DUT_MLP_PERM_FC2[t*C + c] += REF_MLP_B2[l][c];
        }
    }
    //* compare
    compare<int64_t>(REF_MHA_PERM_QKV[l], DUT_MHA_PERM_QKV, H*3*T*HC,  "MHA_PERM_QKV");
    compare<int64_t>(REF_MHA_PERM_O  [l], DUT_MHA_PERM_O,   T*C,       "MHA_PERM_O"  );
    compare<int64_t>(REF_MLP_PERM_FC1[l], DUT_MLP_PERM_FC1, T*CM, "MLP_PERM_FC1");
    compare<int64_t>(REF_MLP_PERM_FC2[l], DUT_MLP_PERM_FC2, T*C,       "MLP_PERM_FC2");
}


// ============================================================================
// test_layer
// ============================================================================
void test_layer(int l_begin, int l_close){
    hls::stream<hls::vector<aq_t, TP *CIP> > i_stream  ("i_stream" );
    hls::stream<hls::vector<as_t, TP     > > s_stream  ("s_stream" );
    hls::stream<hls::vector<wq_t, COP*CIP> > w_stream  ("w_stream" );
    hls::stream<hls::vector<ws_t, COP    > > s1_stream ("s1_stream");
    hls::stream<hls::vector<ws_t, COP    > > s2_stream ("s2_stream");
    hls::stream<hls::vector<of_t, COP    > > o_stream  ("o_stream" );

    for(int l=l_begin; l<l_close; ++l){
        prepare_encoder_data(l, i_stream, s_stream, w_stream, s1_stream, s2_stream);
    }

    top(l_begin, l_close, i_stream, s_stream, w_stream, s1_stream, s2_stream, o_stream);

    assert(i_stream .size() == 0);
    assert(s_stream .size() == 0);
    assert(w_stream .size() == 0);
    assert(s1_stream.size() == 0);
    assert(s2_stream.size() == 0);

    for(int l=l_begin; l<l_close; ++l){
        compare_encoder_data(l, o_stream);
    }
}


int main(){
    test_layer(0, 1);
    // test_layer(0, 2);
    // test_layer(0, VIT_L);
    return 0;
}
