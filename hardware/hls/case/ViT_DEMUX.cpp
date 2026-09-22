#include "../src/common.h"
#include "../src/utils.h"

// ============================================================================
// Vision Encoder (ViT) GEMM_DEMUX
// ============================================================================
// 从 GEMM 输出流中路由到各下游模块:
//   QK → qk_stream, V → v_stream, FC1 → fc1_stream, O/FC2 → od_stream
// 已去掉 CLS (LM Head) 与 Connector 路径


// simulation hyperparameters
constexpr int L             = VIT_L;       // 12
constexpr int T_LOAD        = 5120;
constexpr int POS           = 0;

// model hyperparameters
constexpr int H             = VIT_H;       // 12
constexpr int G             = VIT_G;       // 8
constexpr int C             = VIT_C;       // 768
constexpr int HC            = VIT_HC;      // 64
constexpr int CM            = VIT_MLP_DIM; // 3072

// design hyperparameters
constexpr int T             = 1024;       // 1024 (total tokens, bidirectional)
constexpr int TP            = 8;           // hardware token tile (replaces old T=8)
constexpr int CP            = G;           // = 8

// derived hyperparameters
constexpr int HCT           = HC / CP;     // 8
constexpr int CMT           = CM / CP;     // 384
constexpr int CT            = C  / CP;     // 96

// ViT encoder total output counts (whole T tokens)
const int T_QKVO    = (T * C    ) / CP;
const int T_FC1     = (T * CM   ) / CP;
const int T_FC2     = (T * C    ) / CP;
const int T_DECODER = T_QKVO*4 + T_FC1 + T_FC2;

constexpr int NUM_Y = 3*T*C + T*C + T*CM + T*C;   // QKV+O+FC1+FC2 (all T=1024 tokens)

// TRUNC_BASE = 9 (same as ViT_PERMUTE); encoder paths only
constexpr int TRUNC_BASE = VIT_TRUNC_BASE;



constexpr int VIT_ATTN_LAYER0_Q_BIAS[L][C] = {
    #include "../src/ref/ViT/attn_layer0_bq.txt"
};
constexpr int VIT_ATTN_LAYER0_K_BIAS[L][C] = {
    #include "../src/ref/ViT/attn_layer0_bk.txt"
};
constexpr int VIT_ATTN_LAYER0_V_BIAS[L][C] = {
    #include "../src/ref/ViT/attn_layer0_bv.txt"
};
constexpr int VIT_ATTN_LAYER0_O_BIAS[L][C] = {
    #include "../src/ref/ViT/attn_layer0_bo.txt"
};
constexpr int VIT_MLP_LAYER0_FC1_BIAS[L][CM] = {
    #include "../src/ref/ViT/mlp_layer0_b1.txt"
};
constexpr int VIT_MLP_LAYER0_FC2_BIAS[L][C] = {
    #include "../src/ref/ViT/mlp_layer0_b2.txt"
};

// ============================================================================
// top
// ============================================================================
void top(
    int l_begin,
    int l_close,
    int pos,
    // input: raw GEMM accumulator output
    hls::stream<hls::vector<VIT_GEMM_TRUNC_T,  CP> >& gemm_stream,
    // output streams per path
    hls::stream<hls::vector<VIT_QKV_TRUNC_T,   CP> >& qk_stream,
    hls::stream<hls::vector<VIT_QKV_TRUNC_T,   CP> >& v_stream,
    hls::stream<hls::vector<VIT_FC1_TRUNC_T,   CP> >& fc1_stream,
    hls::stream<hls::vector<VIT_FC2_TRUNC_T,          CP> >& ofc2_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=gemm_stream
    #pragma HLS interface axis port=qk_stream
    #pragma HLS interface axis port=v_stream
    #pragma HLS interface axis port=fc1_stream
    #pragma HLS interface axis port=ofc2_stream
    #pragma HLS aggregate variable=gemm_stream  compact=bit
    #pragma HLS aggregate variable=qk_stream    compact=bit
    #pragma HLS aggregate variable=v_stream     compact=bit
    #pragma HLS aggregate variable=fc1_stream   compact=bit
    #pragma HLS aggregate variable=ofc2_stream  compact=bit


    for(int l=l_begin; l<l_close; ++l){
        if(l < VIT_L){
            //* QKV first
            for(int h=0; h<H; ++h){
                for(int qkv=0; qkv<3; ++qkv){
                    for(int tt=0; tt<T/TP; ++tt){
                        for(int hct=0; hct<HCT; ++hct){
                            for(int t=0; t<TP; ++t){
                                #pragma HLS pipeline II=1
                                hls::vector<VIT_GEMM_TRUNC_T, CP> gemm_vec = gemm_stream.read();
                                hls::vector<VIT_QKV_TRUNC_T, CP> qkv_vec;
                                if(qkv == 0 || qkv == 1){
                                    for(int cp=0; cp<CP; ++cp){
                                        #pragma HLS unroll
                                        qkv_vec[cp] = gemm_vec[cp] >> (VIT_MHA_TRUNC_QK - TRUNC_BASE);
                                        int c_idx = h*HC + hct*CP + cp;
                                        int32_t b = (qkv == 0) ? VIT_ATTN_LAYER0_Q_BIAS[l][c_idx]
                                                               : VIT_ATTN_LAYER0_K_BIAS[l][c_idx];
                                        qkv_vec[cp] += VIT_QKV_TRUNC_T(b);
                                    }
                                    qk_stream.write(qkv_vec);
                                } else {
                                    for(int cp=0; cp<CP; ++cp){
                                        #pragma HLS unroll
                                        qkv_vec[cp] = gemm_vec[cp] >> (VIT_MHA_TRUNC_V - TRUNC_BASE);
                                        int c_idx = h*HC + hct*CP + cp;
                                        qkv_vec[cp] += VIT_QKV_TRUNC_T(VIT_ATTN_LAYER0_V_BIAS[l][c_idx]);
                                    }
                                    v_stream.write(qkv_vec);
                                }
                            }
                        }
                    }
                }
            }
            //* O + FC1 + FC2 by unified iter-style (like LLM_DEMUX)
            for(int iter=3*T_QKVO; iter<T_DECODER; ++iter){
                #pragma HLS pipeline II=1
                hls::vector<VIT_GEMM_TRUNC_T, CP> gemm_vec = gemm_stream.read();
                if(iter < 4*T_QKVO){
                    hls::vector<VIT_FC2_TRUNC_T, CP> o_vec;
                    int rel = iter - 3*T_QKVO;
                    int ct = (rel / TP) % CT;
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        o_vec[cp] = gemm_vec[cp] >> (VIT_MHA_TRUNC_O - TRUNC_BASE);
                        int c_idx = ct*CP + cp;
                        o_vec[cp] += VIT_FC2_TRUNC_T(VIT_ATTN_LAYER0_O_BIAS[l][c_idx]);
                    }
                    ofc2_stream.write(o_vec);
                } else if(iter < 4*T_QKVO + T_FC1){
                    hls::vector<VIT_FC1_TRUNC_T, CP> fc1_vec;
                    int rel = iter - 4*T_QKVO;
                    int cmt = (rel / TP) % CMT;
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        fc1_vec[cp] = gemm_vec[cp] >> (VIT_MLP_TRUNC_FC1 - TRUNC_BASE);
                        int c_idx = cmt*CP + cp;
                        fc1_vec[cp] += VIT_FC1_TRUNC_T(VIT_MLP_LAYER0_FC1_BIAS[l][c_idx]);
                    }
                    fc1_stream.write(fc1_vec);
                } else {
                    hls::vector<VIT_FC2_TRUNC_T, CP> d_vec;
                    int rel = iter - (4*T_QKVO + T_FC1);
                    int ct = (rel / TP) % CT;
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        d_vec[cp] = gemm_vec[cp] >> (VIT_MLP_TRUNC_FC2 - TRUNC_BASE);
                        int c_idx = ct*CP + cp;
                        d_vec[cp] += VIT_FC2_TRUNC_T(VIT_MLP_LAYER0_FC2_BIAS[l][c_idx]);
                    }
                    ofc2_stream.write(d_vec);
                }
            }
        }
    }
}


// ============================================================================
// Reference data
// ============================================================================

int64_t REF_CONDENSED_Y   [L][NUM_Y   ];

int64_t REF_MHA_Q [L][T*C ];
int64_t REF_MHA_K [L][T*C ];
int64_t REF_MHA_V [L][T*C ];
int64_t REF_MHA_O [L][T*C ];
int64_t REF_MLP_FC1[L][T*CM];
int64_t REF_MLP_FC2[L][T*C ];

// permuted refs
int64_t REF_PERM_QK  [L][H  *2*T*HC];
int64_t REF_PERM_V   [L][H    *T*HC];
int64_t REF_PERM_O   [L][H    *T*HC];
int64_t REF_PERM_FC1 [L][CMT  *T*CP];  // single projection (no gate)
int64_t REF_PERM_FC2 [L][      T*C ];

// permuted DUT output
int64_t DUT_PERM_QK  [H  *2*T*HC];
int64_t DUT_PERM_V   [H    *T*HC];
int64_t DUT_PERM_O   [H    *T*HC];
int64_t DUT_PERM_FC1 [CMT  *T*CP];
int64_t DUT_PERM_FC2 [      T*C ];


// ============================================================================
// prepare_encoder_data
// ============================================================================
void prepare_encoder_data(
    int l,
    hls::stream<hls::vector<VIT_GEMM_TRUNC_T, CP> >& gemm_stream
){
    const string binaries_path = VIT_BINARIES_PATH + to_string(l);
    const string condense_path = VIT_CONDENSE_PATH + to_string(l);

    auto CONDENSED_Y = read_tensor<int64_t>(condense_path + "/CONDENSED_GEMM_Y.bin");
    auto MHA_Q       = read_tensor<int64_t>(binaries_path + "/MHA_Q.bin"   );
    auto MHA_K       = read_tensor<int64_t>(binaries_path + "/MHA_K.bin"   );
    auto MHA_V       = read_tensor<int64_t>(binaries_path + "/MHA_V.bin"   );
    auto MHA_O       = read_tensor<int64_t>(binaries_path + "/MHA_O.bin"   );
    auto MLP_XFC1    = read_tensor<int64_t>(binaries_path + "/MLP_XFC1.bin");
    auto MLP_XFC2    = read_tensor<int64_t>(binaries_path + "/MLP_XFC2.bin");

    tensor2array<int64_t>(CONDENSED_Y, REF_CONDENSED_Y[l], 1, 1, 1, 1, NUM_Y, NUM_Y);
    tensor2array<int64_t>(MHA_Q,  REF_MHA_Q  [l], 1, 1, T_LOAD, POS, T, H*HC, H*HC);
    tensor2array<int64_t>(MHA_K,  REF_MHA_K  [l], 1, 1, T_LOAD, POS, T, H*HC, H*HC);
    tensor2array<int64_t>(MHA_V,  REF_MHA_V  [l], 1, 1, T_LOAD, POS, T, H*HC, H*HC);
    tensor2array<int64_t>(MHA_O,  REF_MHA_O  [l], 1, 1, T_LOAD, POS, T, C,    C   );
    tensor2array<int64_t>(MLP_XFC1,REF_MLP_FC1[l],1, 1, T_LOAD, POS, T, CM,   CM  );
    tensor2array<int64_t>(MLP_XFC2,REF_MLP_FC2[l],1, 1, T_LOAD, POS, T, C,    C   );

    //* PERMUTATION
    for(int h=0; h<H; ++h){
        for(int qk=0; qk<2; ++qk){
            for(int t=0; t<T; ++t){
                for(int hc=0; hc<HC; ++hc){
                    REF_PERM_QK[l][h*2*T*HC + qk*T*HC + t*HC + hc] =
                        qk==0 ? REF_MHA_Q[l][t*C + h*HC + hc]
                              : REF_MHA_K[l][t*C + h*HC + hc];
                }
            }
        }
    }
    for(int h=0; h<H; ++h)
        for(int t=0; t<T; ++t)
            for(int hc=0; hc<HC; ++hc)
                REF_PERM_V[l][h*T*HC + t*HC + hc] = REF_MHA_V[l][t*C + h*HC + hc];

    for(int t=0; t<T; ++t)
        for(int c=0; c<C; ++c)
            REF_PERM_O[l][t*C + c] = REF_MHA_O[l][t*C + c];


    for(int t=0; t<T; ++t)
        for(int c=0; c<CM; ++c)
            REF_PERM_FC1[l][t*CM + c] = REF_MLP_FC1[l][t*CM + c];

    for(int t=0; t<T; ++t)
        for(int c=0; c<C; ++c)
            REF_PERM_FC2[l][t*C + c] = REF_MLP_FC2[l][t*C + c];

    array2stream<int64_t, VIT_GEMM_TRUNC_T, 1, 1, 1, 1, NUM_Y, CP>(REF_CONDENSED_Y[l], gemm_stream, "Input Y");
}


// ============================================================================
// compare_encoder_data
// ============================================================================
void compare_encoder_data(
    int l,
    hls::stream<hls::vector<VIT_QKV_TRUNC_T,  CP> >& qk_stream,
    hls::stream<hls::vector<VIT_QKV_TRUNC_T,  CP> >& v_stream,
    hls::stream<hls::vector<VIT_FC1_TRUNC_T,  CP> >& fc1_stream,
    hls::stream<hls::vector<VIT_FC2_TRUNC_T,         CP> >& ofc2_stream
){
    const string condense_path = VIT_CONDENSE_PATH + to_string(l);
    const int NUM_QK   = 2*T*C;    // T=1024 total tokens
    const int NUM_V    =   T*C;
    const int NUM_FC1  =   T*CM;
    const int NUM_OFC2 = 2*T*C;   // O + FC2

    hls::stream<hls::vector<VIT_QKV_TRUNC_T,  CP> > sim_qk_stream  ("sim_qk_stream"  );
    hls::stream<hls::vector<VIT_QKV_TRUNC_T,  CP> > sim_v_stream   ("sim_v_stream"   );
    hls::stream<hls::vector<VIT_FC1_TRUNC_T,  CP> > sim_fc1_stream ("sim_fc1_stream" );
    hls::stream<hls::vector<VIT_FC2_TRUNC_T,         CP> > sim_ofc2_stream("sim_ofc2_stream");

    stream2stream<VIT_QKV_TRUNC_T,  NUM_QK,   CP>(qk_stream,   sim_qk_stream  );
    stream2stream<VIT_QKV_TRUNC_T,  NUM_V,    CP>(v_stream,    sim_v_stream   );
    stream2stream<VIT_FC1_TRUNC_T,  NUM_FC1,  CP>(fc1_stream,  sim_fc1_stream );
    stream2stream<VIT_FC2_TRUNC_T,         NUM_OFC2, CP>(ofc2_stream, sim_ofc2_stream);

    save_condensed_tensor<int64_t, VIT_QKV_TRUNC_T,  2*T*C,  CP>(condense_path + "/CONDENSED_DEMUX_QK.bin",   sim_qk_stream  );
    save_condensed_tensor<int64_t, VIT_QKV_TRUNC_T,  T*C,    CP>(condense_path + "/CONDENSED_DEMUX_V.bin",    sim_v_stream   );
    save_condensed_tensor<int64_t, VIT_FC1_TRUNC_T,  T*CM,   CP>(condense_path + "/CONDENSED_DEMUX_FC1.bin",  sim_fc1_stream );
    save_condensed_tensor<int64_t, VIT_FC2_TRUNC_T,         2*T*C,  CP>(condense_path + "/CONDENSED_DEMUX_OFC2.bin", sim_ofc2_stream);

    stream2array_unpack<int64_t, VIT_QKV_TRUNC_T,  H*2, T, TP, HC,  CP>(sim_qk_stream,   DUT_PERM_QK,  "Output QK",  true);
    stream2array_unpack<int64_t, VIT_QKV_TRUNC_T,  H,   T, TP, HC,  CP>(sim_v_stream,    DUT_PERM_V,   "Output V",   true);
    stream2array_unpack<int64_t, VIT_FC2_TRUNC_T,  1,   T, TP, C,   CP>(sim_ofc2_stream, DUT_PERM_O,   "Output O",   true);
    stream2array_unpack<int64_t, VIT_FC1_TRUNC_T,  1,   T, TP, CM,  CP>(sim_fc1_stream,  DUT_PERM_FC1, "Output FC1", true);
    stream2array_unpack<int64_t, VIT_FC2_TRUNC_T,  1,   T, TP, C,   CP>(sim_ofc2_stream, DUT_PERM_FC2, "Output FC2", true);

    // Q/K/V refs contain bias while GEMM output does not.
    // Calibrate per-output-channel bias from t=0 and add back to all tokens.

    compare<int64_t>(REF_PERM_QK [l], DUT_PERM_QK,  H  *2*T*HC, "QK" );
    compare<int64_t>(REF_PERM_V  [l], DUT_PERM_V,   H    *T*HC, "V"  );
    compare<int64_t>(REF_PERM_O  [l], DUT_PERM_O,   T*C,        "O"  );
    compare<int64_t>(REF_PERM_FC1[l], DUT_PERM_FC1, CMT  *T*CP, "FC1");
    compare<int64_t>(REF_PERM_FC2[l], DUT_PERM_FC2, T*C,        "FC2");
}


// ============================================================================
// test_layer
// ============================================================================
void test_layer(int l_begin, int l_close){
    hls::stream<hls::vector<VIT_GEMM_TRUNC_T,  CP> > gemm_stream  ("gemm_stream"  );
    hls::stream<hls::vector<VIT_QKV_TRUNC_T,   CP> > qk_stream    ("qk_stream"    );
    hls::stream<hls::vector<VIT_QKV_TRUNC_T,   CP> > v_stream     ("v_stream"     );
    hls::stream<hls::vector<VIT_FC1_TRUNC_T,   CP> > fc1_stream   ("fc1_stream"   );
    hls::stream<hls::vector<VIT_FC2_TRUNC_T,          CP> > ofc2_stream  ("ofc2_stream"  );

    for(int l=l_begin; l<l_close; ++l){
        if(l < VIT_L)
            prepare_encoder_data(l, gemm_stream);
    }

    top(l_begin, l_close, POS, gemm_stream, qk_stream, v_stream, fc1_stream, ofc2_stream);

    assert(gemm_stream.size() == 0);

    for(int l=l_begin; l<l_close; ++l){
        if(l < VIT_L)
            compare_encoder_data(l, qk_stream, v_stream, fc1_stream, ofc2_stream);
    }
}


int main(){
    test_layer(0, 1);
    return 0;
}
