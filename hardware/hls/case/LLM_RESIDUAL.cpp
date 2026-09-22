#include "../src/common.h"
#include "../src/utils.h"

// ============================================================================
// LLM_RESIDUAL：Pre-LN 下的残差与流式接口（阅读 top 前建议通读本节）
// ============================================================================
//
// 【残差在算什么】
//   Pre-LN 典型形式：x_new = x + F(RMSNorm(x))。分支算的是增量 Δ；参与加法的主路径是「当前的 x」。
//   不是把历史上所有层的 Δ 无层间变换地堆进一个累加器；而是每层（每 pass）对「当时的 x」加「这一段的 Δ」，
//   结果写回 x_buf，再作为下一 pass / 下一层的输入。
//
// 【本模块在数据流里干什么（两路分工）】
//   - 一路经 res_o_stream 送给下游（RMSNorm → MHA 或 MLP 等），得到 Δ 后经 res_i_stream 回来。
//   - 另一路：同一份 x 留在 x_buf 里，等 Δ 到齐后做 x_buf <- x + Δ（原地更新）。
//   因此「先 write res_o、再 read res_i 相加」是刻意的：下游必须先拿到 x 才能算 F(LN(x))，加法必然晚于 res_o。
//
// 【接口与打包】
//   - x_stream：本段 top 调用开头灌入初始隐状态（TP*CP / beat，tt 外 ct 内，与 res_o 一致）。
//   - res_o_stream：每次 pass 打出「本段残差加法之前」的 x（pack，tt 外 ct 内），供 RMSNorm 与子层；即 Pre-LN 里进 LN 的那份 x。
//   - res_i_stream：分支输出 Δ（unpack，ct 外 tt 内，与 OD/GEMM 顺序对齐），宽度 CP，每块读 TP 次拼成 TP*CP。
//   - y_stream：仅在整次 top 末尾写一次——整段 [l_begin,l_close) 层全部跑完后 x_buf 的最终 hidden（非 CLS）。
//     中间层不向 y_stream 写；层与层之间状态只在 x_buf 内传递。若整网分多次 top，需片外把上次输出再接回 x_stream。
//   - l_begin / l_close：处理层索引 l ∈ [l_begin, l_close)，且要求 l < LLAMA_L 才会进入层循环体。
//
// 【每层循环】
//   pass==0：MHA 残差（golden 里 res_o 对应 MHA_X）；pass==1：MLP 残差（res_o 对应 MLP_X）。
//
// 【CLS】
//   run_cls = (l_close == LLAMA_L+1) 时，末尾不写 y_stream，而写 res_o_stream，便于再接分类等路径。
//
// 【仿真 compare_decoder_data】
//   从 res_o_stream 按层消费 2*T*C，与 REF_RESIDUAL_O（前半 MHA_X、后半 MLP_X）比：即「每次进该层 RMSNorm 的 x」是否正确。
//   y_stream 仅在 l == l_close-1 时与 REF_Y（MLP_XD_RES）比：本段最后一层、MLP 残差后的 hidden。
//
// 【test_layer(LLAMA_L, LLAMA_L+1) 注意】
//   层循环条件为 l < l_close && l < LLAMA_L，故 l_begin==LLAMA_L 时层循环一次都不执行：只有 x_stream 灌 buf + 末尾 CLS 走 res_o，
//   并非「跑完最后一层 Decoder 再 CLS」的完整语义。
//
// 【原设计 consideration 摘要】
//   1. 并行度：res_i unpack（GEMM/OD），res_o pack（去 RMSNorm）。
//   2. CLS：最终 hidden 走 res_o 而非 y。
//   3. 预处理/后处理：先 x → x_buf；层内反复 res_o / res_i；最后 x_buf → y 或 res_o。
//   4. 每层两次残差：先整 buf 写 res_o，再读 res_i 加回 x_buf。
// ============================================================================

// simulation hyperparameters
constexpr int L         = LLAMA_L;
constexpr int T_LOAD    = 612;  // saved seq length
constexpr int POS       = 96;   // random position

// model hyperparameters
constexpr int C         = LLAMA_C;
constexpr int G         = LLAMA_G;

// design hyperparameters
constexpr int T         = 8;
constexpr int TP        = 1;

// derived hyperparameters
constexpr int TT        = T / TP;
constexpr int CP        = G;
constexpr int CT        = C / CP;
constexpr int NUM_X     = T*C;
constexpr int NUM_OD    = T*C;

// dtypes: X_T
typedef X_T X_T;     // largest accumulated value

// top：一次调用内 x_buf 持有当前隐状态；res_o 输出「加 Δ 前」的 x；res_i 输入 Δ；末尾一次性输出最终 hidden（y 或 CLS 时 res_o）。
void top(
    // scalar inputs
    int     l_begin,
    int     l_close,
    // streams
    hls::stream<hls::vector<X_T, TP*CP> >& x_stream,
    hls::stream<hls::vector<X_T,    CP> >& res_i_stream,
    hls::stream<hls::vector<X_T, TP*CP> >& res_o_stream,
    hls::stream<hls::vector<X_T, TP*CP> >& y_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    // set interface
    #pragma HLS interface axis port=x_stream
    #pragma HLS interface axis port=res_i_stream
    #pragma HLS interface axis port=res_o_stream
    #pragma HLS interface axis port=y_stream
    // set aggregate pragma
    #pragma HLS aggregate variable=x_stream         compact=bit
    #pragma HLS aggregate variable=res_i_stream     compact=bit
    #pragma HLS aggregate variable=res_o_stream     compact=bit
    #pragma HLS aggregate variable=y_stream         compact=bit

    X_T x_buf[T][C];
    #pragma HLS array_reshape variable=x_buf cyclic factor=TP dim=1
    #pragma HLS array_reshape variable=x_buf cyclic factor=CP dim=2
    #pragma HLS bind_storage  variable=x_buf type=RAM_2P impl=URAM

    // 流程：预处理 x→x_buf → [每层: res_o 打出 x → res_i 加回]×2 pass → 后处理 x_buf→y 或 res_o（CLS）

    // 预处理：本段第一次的隐状态从 x_stream 按 pack 顺序写入 x_buf（仅调用开头一次）
    for(int tt=0; tt<TT; ++tt){
        for(int ct=0; ct<CT; ++ct){
            #pragma HLS pipeline II=1
            hls::vector<X_T, TP*CP> vec = x_stream.read();
            for(int tp=0; tp<TP; ++tp){
                for(int cp=0; cp<CP; ++cp){
                    #pragma HLS unroll
                    x_buf[tt*TP + tp][ct*CP + cp] = vec[tp*CP + cp];
                }
            }
        }
    }

    // 主体：每层 pass0=MHA 残差、pass1=MLP 残差（层与层之间只靠 x_buf 递推，不写 y_stream）
    for(int l=l_begin; l<l_close && l<LLAMA_L; ++l){
        for(int pass=0; pass<2; ++pass){
            // 本 pass：先把「加 Δ 前」的 x_buf 整表打出给 RMSNorm/子层（与下游约定的 pack 顺序：tt 外 ct 内）
            for(int tt=0; tt<TT; ++tt){
                for(int ct=0; ct<CT; ++ct){
                    #pragma HLS pipeline II=1
                    hls::vector<X_T, TP*CP> vec;
                    for(int tp=0; tp<TP; ++tp){
                        for(int cp=0; cp<CP; ++cp){
                            #pragma HLS unroll
                            vec[tp*CP + cp] = x_buf[tt*TP + tp][ct*CP + cp];
                        }
                    }
                    res_o_stream.write(vec);
                }
            }
            // 子层算完：按 unpack 顺序（ct 外 tt 内）读 res_i 得到 Δ，与当前 x_buf 相加写回 → 即本 pass 的 x_new
            for(int ct=0; ct<CT; ++ct){
                for(int tt=0; tt<TT; ++tt){
                    #pragma HLS pipeline II=1
                    // declare vec
                    hls::vector<X_T, TP*CP> vec_i;
                    hls::vector<X_T, TP*CP> vec_buf;
                    // read unpacked tokens
                    for(int tp=0; tp<TP; ++tp){
                        hls::vector<X_T, CP> vec_cp = res_i_stream.read();
                        for(int cp=0; cp<CP; ++cp){
                            vec_i[tp*CP + cp] = vec_cp[cp];
                        }
                    }
                    // read from buffer
                    for(int tp=0; tp<TP; ++tp){
                        for(int cp=0; cp<CP; ++cp){
                            vec_buf[tp*CP + cp] = x_buf[tt*TP + tp][ct*CP + cp];
                        }
                    }
                    // add to buffer
                    for(int tp=0; tp<TP; ++tp){
                        for(int cp=0; cp<CP; ++cp){
                            x_buf[tt*TP + tp][ct*CP + cp] = vec_i[tp*CP + cp] + vec_buf[tp*CP + cp];
                        }
                    }
                }
            }
        }
    }

    // 后处理：整次调用仅此处输出「最终」hidden。非 CLS→y_stream（供后续 final norm / lm_head 等）；CLS→res_o_stream
    bool run_cls = (l_close == LLAMA_L+1);
    for(int tt=0; tt<TT; ++tt){
        for(int ct=0; ct<CT; ++ct){
            #pragma HLS pipeline II=1
            hls::vector<X_T, TP*CP> vec;
            for(int tp=0; tp<TP; ++tp){
                for(int cp=0; cp<CP; ++cp){
                    #pragma HLS unroll
                    vec[tp*CP + cp] = x_buf[tt*TP + tp][ct*CP + cp];
                }
            }
            if(run_cls)    res_o_stream    .write(vec);
            else           y_stream        .write(vec);
        }
    }
}

// declare the refs
int64_t REF_X               [L][  T*C  ];
int64_t REF_Y               [L][  T*C  ];
int64_t REF_MHA_X           [L][  T*C  ];
int64_t REF_MHA_O           [L][  T*C  ];
int64_t REF_MLP_X           [L][  T*C  ];
int64_t REF_MLP_D           [L][  T*C  ];
int64_t REF_MHA_O_RES       [L][  T*C  ];
int64_t REF_MLP_XD_RES      [L][  T*C  ];
// declare the condensed input
int64_t REF_CONDENSED_OD    [L][2*NUM_X];
// ref output：每层 golden 的 res_o 共 2*T*C（pass0 MHA_X + pass1 MLP_X），与 DUT 比对用 2*T*C
int64_t REF_RESIDUAL_O      [L][3*T*C  ];
// declare the ref residual
int64_t DUT_Y                  [  T*C  ];
int64_t DUT_RESIDUAL_O      [L][3*T*C  ];


void prepare_decoder_data(
    int l,
    int l_begin,
    int l_close,
    const string& binaries_path,
    const string& condense_path,
    hls::stream<hls::vector<X_T, TP*CP> >& x_stream,
    hls::stream<hls::vector<X_T,    CP> >& res_i_stream
){
    //* condensed input res_i_stream
    auto CONDENSED_OD   = read_tensor<int64_t> (condense_path + "/CONDENSED_DEMUX_OD.bin");
    tensor2array<int64_t>(  CONDENSED_OD,   REF_CONDENSED_OD[l],    1,      1,      1,          1,      2*NUM_X,    2*NUM_X);

    //* read results
    auto MHA_X          = read_tensor<int64_t> (binaries_path + "/MHA_X.bin"     );
    auto MHA_O          = read_tensor<int64_t> (binaries_path + "/MHA_O.bin"     );
    auto MHA_O_RES      = read_tensor<int64_t> (binaries_path + "/MHA_O_RES.bin" );
    auto MLP_X          = read_tensor<int64_t> (binaries_path + "/MLP_X.bin"     );
    auto MLP_XD         = read_tensor<int64_t> (binaries_path + "/MLP_XD.bin"    );
    auto MLP_XD_RES     = read_tensor<int64_t> (binaries_path + "/MLP_XD_RES.bin");

    //             dtype,   tensor,         array,                      H_LOAD, H,      T_LOAD,     T_START,    T,      C_LOAD,     C
    tensor2array<int64_t>(  MHA_X,          REF_MHA_X       [l],        1,      1,      T_LOAD,     POS,        T,      C,          C      );
    tensor2array<int64_t>(  MHA_O,          REF_MHA_O       [l],        1,      1,      T_LOAD,     POS,        T,      C,          C      );
    tensor2array<int64_t>(  MLP_X,          REF_MLP_X       [l],        1,      1,      T_LOAD,     POS,        T,      C,          C      );
    tensor2array<int64_t>(  MLP_XD,         REF_MLP_D       [l],        1,      1,      T_LOAD,     POS,        T,      C,          C      );
    tensor2array<int64_t>(  MHA_O_RES,      REF_MHA_O_RES   [l],        1,      1,      T_LOAD,     POS,        T,      C,          C      );
    tensor2array<int64_t>(  MLP_XD_RES,     REF_MLP_XD_RES  [l],        1,      1,      T_LOAD,     POS,        T,      C,          C      );

    //* check the coherency of the input itself
    for(int t=0; t<T; ++t){
        for(int c=0; c<C; ++c){
            // MHA_X + MHA_O -> MLP_X
            assert(REF_MHA_X[l][t*C + c] + REF_MHA_O[l][t*C + c] == REF_MHA_O_RES [l][t*C + c]);
            // MLP_X + MLP_XD -> MLP_XD_RES
            assert(REF_MLP_X[l][t*C + c] + REF_MLP_D[l][t*C + c] == REF_MLP_XD_RES[l][t*C + c]);
        }
    }

    //* put REF_MHA_X and REF_MLP_X into the ref residual
    for(int pass=0; pass<2; ++pass){
        for(int t=0; t<T; ++t){
            for(int c=0; c<C; ++c){
                if(pass == 0) REF_RESIDUAL_O[l][pass*T*C + t*C + c] = REF_MHA_X[l][t*C + c];
                else          REF_RESIDUAL_O[l][pass*T*C + t*C + c] = REF_MLP_X[l][t*C + c];
            }
        }
    }

    //* put MLP_XD_RES into the ref y
    for(int t=0; t<T; ++t){
        for(int c=0; c<C; ++c){
            REF_Y[l][t*C + c] = REF_MLP_XD_RES[l][t*C + c];
        }
    }

    //* write stream
    array2stream<int64_t, X_T, 1, 1, 1, 1, 2*NUM_X, CP>(REF_CONDENSED_OD[l], res_i_stream, "CONDENSED OD", true);

    //* save condensed input
    save_condensed_tensor<int64_t, X_T, 2*NUM_X, CP>(condense_path + "/CONDENSED_RESIDUAL_I.bin", res_i_stream);

    //* stream in first layer
    if(l == l_begin){
        for(int t=0; t<T; ++t){
            for(int c=0; c<C; ++c){
                REF_X[l][t*C + c] = REF_MHA_X[l_begin][t*C + c];
            }
        }
        array2stream<int64_t, X_T, 1, 1, T, TP, C, CP>(REF_X[l],     x_stream,     "X",            true);
        //* save condensed input
        save_condensed_tensor<int64_t, X_T, T*C, TP*CP>(condense_path + "/CONDENSED_RESIDUAL_X.bin", x_stream);
    }
}


// 校验：从 res_o_stream 按顺序剥每层 2*T*C，与 REF_RESIDUAL_O 比（Pre-LN 下即该层两次进 RMSNorm 的 x）。
// 仅当 l==l_close-1 时读 y_stream 与 REF_Y（MLP_XD_RES）比——对应本段最后一次 top 末尾写出的最终 hidden。
void compare_decoder_data(
    int l,
    int l_begin,
    int l_close,
    const string& binaries_path,
    const string& condense_path,
    hls::stream<hls::vector<X_T, TP*CP> >& res_o_stream,
    hls::stream<hls::vector<X_T, TP*CP> >& y_stream
){
    //* save condensed output
    hls::stream<hls::vector<X_T, TP*CP> > sim_res_o_stream("sim_res_o_stream");
    stream2stream                 <X_T, 2*T*C, TP*CP>(res_o_stream, sim_res_o_stream);
    save_condensed_tensor<int64_t, X_T, 2*T*C, TP*CP>(condense_path + "/CONDENSED_RESIDUAL_O.bin", sim_res_o_stream);
    //* readout
    stream2array<int64_t, X_T, 2, T, TP, C, CP>(sim_res_o_stream, DUT_RESIDUAL_O[l], "RESIDUAL O", true);
    //* check stream size
    assert(sim_res_o_stream.size() == 0);
    //* compare residual
    compare<int64_t>(REF_RESIDUAL_O[l], DUT_RESIDUAL_O[l], 2*T*C, "RESIDUAL");

    //* compare y
    if(l == l_close-1){
        hls::stream<hls::vector<X_T, TP*CP> > sim_y_stream("sim_y_stream");
        stream2stream                 <X_T, T*C, TP*CP>(y_stream, sim_y_stream);
        save_condensed_tensor<int64_t, X_T, T*C, TP*CP>(condense_path + "/CONDENSED_RESIDUAL_Y.bin", sim_y_stream);
        stream2array<int64_t, X_T, 1, T, TP, C, CP>(sim_y_stream, DUT_Y, "Y", true);
        compare<int64_t>(REF_Y[l], DUT_Y, T*C, "Y");
    }
}


// 对 [l_begin,l_close) 每层 prepare（堆叠 res_i；仅 l==l_begin 时推 x_stream），再单次调用 top，再逐层 compare。
void test_layer(int l_begin, int l_close){
    //* create streams
    hls::stream<hls::vector<X_T, TP*CP> > x_stream       ("X");
    hls::stream<hls::vector<X_T,    CP> > res_i_stream   ("RESIDUAL I");
    hls::stream<hls::vector<X_T, TP*CP> > res_o_stream   ("RESIDUAL O");
    hls::stream<hls::vector<X_T, TP*CP> > y_stream       ("Y");


    //* for each layer, prepare the data
    for(int l=l_begin; l<l_close && l<LLAMA_L; ++l){
        prepare_decoder_data(
            l,
            l_begin,
            l_close,
            BINARIES_PATH + to_string(l),
            CONDENSE_PATH + to_string(l),
            x_stream,
            res_i_stream
        );
    }


    //* call top function
    top(l_begin, l_close, x_stream, res_i_stream, res_o_stream, y_stream);


    //* for each layer, compare the residual data
    for(int l=l_begin; l<l_close && l<LLAMA_L; ++l){
        compare_decoder_data(
            l,
            l_begin,
            l_close,
            BINARIES_PATH + to_string(l),
            CONDENSE_PATH + to_string(l),
            res_o_stream,
            y_stream
        );
    }

}


int main(){
    // 可改测：test_layer(0, LLAMA_L) 多层；test_layer(l,l+1) 单层；test_layer(LLAMA_L, LLAMA_L+1) 见文件头「CLS 注意」（层循环体可能为零次）
    for(int l=0; l<1; ++l){
        test_layer(l, l+1);
        printf("Layer %d passed\n", l);
    }

    return 0;
}