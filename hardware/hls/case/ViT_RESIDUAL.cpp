#include "../src/common.h"
#include "../src/utils.h"

// ============================================================================
// ViT_RESIDUAL：Pre-LN 下的残差与流式接口（对应 LLM_RESIDUAL）
// ============================================================================
// 与 LLM 的区别：
//   - 残差类型 VIT_X_T (int23) 代替 X_T (int25)
//   - T=1024（全部 image patch），TP=1（每拍 1 token）
//   - x_buf[T=1024][C=768] 存于 URAM（~6 MB）
//   - 每层两次残差：MHA 后（x += O），MLP 后（x += FC2）
//   - 输入 res_i_stream 来自 CONDENSED_DEMUX_OFC2.bin（O 和 FC2 交错）
//   - 输出 CONDENSED_RESIDUAL_O.bin（VIT_X_T）供 ViT_LAYERNORM 使用
// ============================================================================

// simulation hyperparameters
constexpr int L         = VIT_L;    // 12
constexpr int T_LOAD    = 5120;     // saved seq length (batch * patches)
constexpr int POS       = 0;        // start position (取第一张图的 1024 patches)

// model hyperparameters
constexpr int C         = VIT_C;    // 768
constexpr int G         = VIT_G;    // 8

// design hyperparameters
constexpr int T         = VIT_S;    // 1024
constexpr int TP        = 1;        // 每拍 1 token（非 MUX/RV_GEMM 无需 TP>1）

// derived hyperparameters
constexpr int TT        = T / TP;   // 1024
constexpr int CP        = G;        // 8
constexpr int CT        = C / CP;   // 96
constexpr int NUM_X     = T * C;    // 786,432
// DEMUX res_i_stream 顺序为 TT_D=128→CT=96→DEMUX_TP=8（来自 CONDENSED_DEMUX_OFC2.bin）
constexpr int DEMUX_TP  = 8;
constexpr int TT_D      = T / DEMUX_TP;  // 128

// Pack the flat T*C residual array as 3 x 23-bit elements per 69-bit word.
// A cyclic factor=16 partition gives sixteen independent 69-bit URAM banks:
// ceil((T*C/3)/16 / 4096) * 16 = 4 * 16 = 64 URAM, with no BRAM spill.
constexpr int X_PACK       = 3;
constexpr int X_BUF_WORDS  = (NUM_X + X_PACK - 1) / X_PACK;
constexpr int X_BUF_BANKS  = 16;
static_assert(C % X_PACK == 0, "C must be divisible by X_PACK");

typedef ap_uint<VIT_DW_X * X_PACK> vit_x_pack_t;

static ap_uint<VIT_DW_X> vit_x_to_bits(VIT_X_T val){
    #pragma HLS inline
    ap_uint<VIT_DW_X> bits = val.range(VIT_DW_X - 1, 0);
    return bits;
}

static VIT_X_T vit_x_from_bits(ap_uint<VIT_DW_X> bits){
    #pragma HLS inline
    VIT_X_T val;
    val.range(VIT_DW_X - 1, 0) = bits;
    return val;
}

static VIT_X_T get_pack_lane(vit_x_pack_t pack, int lane){
    #pragma HLS inline
    switch(lane){
        case 0: return vit_x_from_bits(pack.range(1 * VIT_DW_X - 1, 0 * VIT_DW_X));
        case 1: return vit_x_from_bits(pack.range(2 * VIT_DW_X - 1, 1 * VIT_DW_X));
        default: return vit_x_from_bits(pack.range(3 * VIT_DW_X - 1, 2 * VIT_DW_X));
    }
}

static void set_pack_lane(vit_x_pack_t& pack, int lane, VIT_X_T val){
    #pragma HLS inline
    switch(lane){
        case 0: pack.range(1 * VIT_DW_X - 1, 0 * VIT_DW_X) = vit_x_to_bits(val); break;
        case 1: pack.range(2 * VIT_DW_X - 1, 1 * VIT_DW_X) = vit_x_to_bits(val); break;
        default: pack.range(3 * VIT_DW_X - 1, 2 * VIT_DW_X) = vit_x_to_bits(val); break;
    }
}

static int x_base_word(int token, int ct){
    #pragma HLS inline
    return token * (C / X_PACK) + (ct * CP) / X_PACK;
}

static int x_base_offset(int ct){
    #pragma HLS inline
    return (ct * CP) % X_PACK;
}

static void unpack_x_words(
    vit_x_pack_t pack0,
    vit_x_pack_t pack1,
    vit_x_pack_t pack2,
    vit_x_pack_t pack3,
    int base_off,
    hls::vector<VIT_X_T, CP>& vec
){
    #pragma HLS inline
    if(base_off == 0){
        vec[0] = get_pack_lane(pack0, 0);
        vec[1] = get_pack_lane(pack0, 1);
        vec[2] = get_pack_lane(pack0, 2);
        vec[3] = get_pack_lane(pack1, 0);
        vec[4] = get_pack_lane(pack1, 1);
        vec[5] = get_pack_lane(pack1, 2);
        vec[6] = get_pack_lane(pack2, 0);
        vec[7] = get_pack_lane(pack2, 1);
    }else if(base_off == 1){
        vec[0] = get_pack_lane(pack0, 1);
        vec[1] = get_pack_lane(pack0, 2);
        vec[2] = get_pack_lane(pack1, 0);
        vec[3] = get_pack_lane(pack1, 1);
        vec[4] = get_pack_lane(pack1, 2);
        vec[5] = get_pack_lane(pack2, 0);
        vec[6] = get_pack_lane(pack2, 1);
        vec[7] = get_pack_lane(pack2, 2);
    }else{
        vec[0] = get_pack_lane(pack0, 2);
        vec[1] = get_pack_lane(pack1, 0);
        vec[2] = get_pack_lane(pack1, 1);
        vec[3] = get_pack_lane(pack1, 2);
        vec[4] = get_pack_lane(pack2, 0);
        vec[5] = get_pack_lane(pack2, 1);
        vec[6] = get_pack_lane(pack2, 2);
        vec[7] = get_pack_lane(pack3, 0);
    }
}

static void pack_x_words(
    const hls::vector<VIT_X_T, CP>& vec,
    int base_off,
    vit_x_pack_t& pack0,
    vit_x_pack_t& pack1,
    vit_x_pack_t& pack2,
    vit_x_pack_t& pack3
){
    #pragma HLS inline
    if(base_off == 0){
        set_pack_lane(pack0, 0, vec[0]);
        set_pack_lane(pack0, 1, vec[1]);
        set_pack_lane(pack0, 2, vec[2]);
        set_pack_lane(pack1, 0, vec[3]);
        set_pack_lane(pack1, 1, vec[4]);
        set_pack_lane(pack1, 2, vec[5]);
        set_pack_lane(pack2, 0, vec[6]);
        set_pack_lane(pack2, 1, vec[7]);
    }else if(base_off == 1){
        set_pack_lane(pack0, 1, vec[0]);
        set_pack_lane(pack0, 2, vec[1]);
        set_pack_lane(pack1, 0, vec[2]);
        set_pack_lane(pack1, 1, vec[3]);
        set_pack_lane(pack1, 2, vec[4]);
        set_pack_lane(pack2, 0, vec[5]);
        set_pack_lane(pack2, 1, vec[6]);
        set_pack_lane(pack2, 2, vec[7]);
    }else{
        set_pack_lane(pack0, 2, vec[0]);
        set_pack_lane(pack1, 0, vec[1]);
        set_pack_lane(pack1, 1, vec[2]);
        set_pack_lane(pack1, 2, vec[3]);
        set_pack_lane(pack2, 0, vec[4]);
        set_pack_lane(pack2, 1, vec[5]);
        set_pack_lane(pack2, 2, vec[6]);
        set_pack_lane(pack3, 0, vec[7]);
    }
}

// top function
void top(
    // scalar inputs
    int     l_begin,
    int     l_close,
    // streams
    hls::stream<hls::vector<VIT_X_T, TP*CP> >& x_stream,
    hls::stream<hls::vector<VIT_X_T,    CP> >& res_i_stream,
    hls::stream<hls::vector<VIT_X_T, TP*CP> >& res_o_stream,
    hls::stream<hls::vector<VIT_X_T, TP*CP> >& y_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=x_stream
    #pragma HLS interface axis port=res_i_stream
    #pragma HLS interface axis port=res_o_stream
    #pragma HLS interface axis port=y_stream
    #pragma HLS aggregate variable=x_stream         compact=bit
    #pragma HLS aggregate variable=res_i_stream     compact=bit
    #pragma HLS aggregate variable=res_o_stream     compact=bit
    #pragma HLS aggregate variable=y_stream         compact=bit

    vit_x_pack_t x_buf[X_BUF_WORDS];
    #pragma HLS array_partition variable=x_buf cyclic factor=X_BUF_BANKS dim=1
    #pragma HLS bind_storage variable=x_buf type=RAM_2P impl=URAM

    // 预处理：从 x_stream 按 pack 顺序写入 x_buf（仅首次调用）
    for(int tt=0; tt<TT; ++tt){
        for(int ct=0; ct<CT; ++ct){
            #pragma HLS pipeline II=1
            hls::vector<VIT_X_T, TP*CP> vec = x_stream.read();
            for(int tp=0; tp<TP; ++tp){
                hls::vector<VIT_X_T, CP> lanes;
                int token = tt*TP + tp;
                int base_word = x_base_word(token, ct);
                int base_off  = x_base_offset(ct);
                vit_x_pack_t pack0 = x_buf[base_word + 0];
                vit_x_pack_t pack1 = x_buf[base_word + 1];
                vit_x_pack_t pack2 = x_buf[base_word + 2];
                vit_x_pack_t pack3 = 0;
                if(base_off == 2) pack3 = x_buf[base_word + 3];
                for(int cp=0; cp<CP; ++cp){
                    #pragma HLS unroll
                    lanes[cp] = vec[tp*CP + cp];
                }
                pack_x_words(lanes, base_off, pack0, pack1, pack2, pack3);
                x_buf[base_word + 0] = pack0;
                x_buf[base_word + 1] = pack1;
                x_buf[base_word + 2] = pack2;
                if(base_off == 2) x_buf[base_word + 3] = pack3;
            }
        }
    }

    // 主体：每层 pass0=MHA 残差、pass1=MLP 残差
    for(int l=l_begin; l<l_close && l<VIT_L; ++l){
        for(int pass=0; pass<2; ++pass){
            // 将「加 Δ 前」的 x_buf 打出给 LayerNorm（pack 顺序：tt 外 ct 内）
            for(int tt=0; tt<TT; ++tt){
                for(int ct=0; ct<CT; ++ct){
                    #pragma HLS pipeline II=1
                    hls::vector<VIT_X_T, TP*CP> vec;
                    for(int tp=0; tp<TP; ++tp){
                        hls::vector<VIT_X_T, CP> lanes;
                        int token = tt*TP + tp;
                        int base_word = x_base_word(token, ct);
                        int base_off  = x_base_offset(ct);
                        vit_x_pack_t pack0 = x_buf[base_word + 0];
                        vit_x_pack_t pack1 = x_buf[base_word + 1];
                        vit_x_pack_t pack2 = x_buf[base_word + 2];
                        vit_x_pack_t pack3 = 0;
                        if(base_off == 2) pack3 = x_buf[base_word + 3];
                        unpack_x_words(pack0, pack1, pack2, pack3, base_off, lanes);
                        for(int cp=0; cp<CP; ++cp){
                            #pragma HLS unroll
                            vec[tp*CP + cp] = lanes[cp];
                        }
                    }
                    res_o_stream.write(vec);
                }
            }
            // 从 res_i_stream 读 Δ（DEMUX 顺序：TT_D=128 外→CT=96 中→DEMUX_TP=8 内），加回 x_buf
            for(int tt_d=0; tt_d<TT_D; ++tt_d){
                for(int ct=0; ct<CT; ++ct){
                    for(int tp=0; tp<DEMUX_TP; ++tp){
                        #pragma HLS pipeline II=1
                        hls::vector<VIT_X_T, CP> vec_cp = res_i_stream.read();
                        int token = tt_d * DEMUX_TP + tp;
                        hls::vector<VIT_X_T, CP> lanes;
                        int base_word = x_base_word(token, ct);
                        int base_off  = x_base_offset(ct);
                        vit_x_pack_t pack0 = x_buf[base_word + 0];
                        vit_x_pack_t pack1 = x_buf[base_word + 1];
                        vit_x_pack_t pack2 = x_buf[base_word + 2];
                        vit_x_pack_t pack3 = 0;
                        if(base_off == 2) pack3 = x_buf[base_word + 3];
                        unpack_x_words(pack0, pack1, pack2, pack3, base_off, lanes);
                        for(int cp=0; cp<CP; ++cp){
                            #pragma HLS unroll
                            lanes[cp] += vec_cp[cp];
                        }
                        pack_x_words(lanes, base_off, pack0, pack1, pack2, pack3);
                        x_buf[base_word + 0] = pack0;
                        x_buf[base_word + 1] = pack1;
                        x_buf[base_word + 2] = pack2;
                        if(base_off == 2) x_buf[base_word + 3] = pack3;
                    }
                }
            }
        }
    }

    // 后处理：将最终 x_buf 写入 y_stream
    for(int tt=0; tt<TT; ++tt){
        for(int ct=0; ct<CT; ++ct){
            #pragma HLS pipeline II=1
            hls::vector<VIT_X_T, TP*CP> vec;
            for(int tp=0; tp<TP; ++tp){
                hls::vector<VIT_X_T, CP> lanes;
                int token = tt*TP + tp;
                int base_word = x_base_word(token, ct);
                int base_off  = x_base_offset(ct);
                vit_x_pack_t pack0 = x_buf[base_word + 0];
                vit_x_pack_t pack1 = x_buf[base_word + 1];
                vit_x_pack_t pack2 = x_buf[base_word + 2];
                vit_x_pack_t pack3 = 0;
                if(base_off == 2) pack3 = x_buf[base_word + 3];
                unpack_x_words(pack0, pack1, pack2, pack3, base_off, lanes);
                for(int cp=0; cp<CP; ++cp){
                    #pragma HLS unroll
                    vec[tp*CP + cp] = lanes[cp];
                }
            }
            y_stream.write(vec);
        }
    }
}

// 声明参考与 DUT 数组（单层，不含 L 维：避免 L*T*C 过大的静态内存）
int64_t REF_X          [T*C];
int64_t REF_MHA_X      [T*C];
int64_t REF_MHA_O      [T*C];
int64_t REF_MLP_X      [T*C];
int64_t REF_MLP_FC2    [T*C];
int64_t REF_MHA_O_RES  [T*C];
int64_t REF_MLP_FC2_RES[T*C];
// condensed 输入（O + FC2 交错，共 2 次 pass）
int64_t REF_CONDENSED_OFC2[2*NUM_X];
// ref res_o（每层 2 次 pass 的 x：MHA_X + MLP_X）
int64_t REF_RESIDUAL_O    [2*T*C];
// DUT 输出
int64_t DUT_Y              [T*C];
int64_t DUT_RESIDUAL_O    [2*T*C];


void prepare_encoder_data(
    int l,
    int l_begin,
    int l_close,
    const string& binaries_path,
    const string& condense_path,
    hls::stream<hls::vector<VIT_X_T, TP*CP> >& x_stream,
    hls::stream<hls::vector<VIT_X_T,    CP> >& res_i_stream
){
    //* condensed input res_i_stream (ViT_DEMUX 的 O + FC2 输出)
    auto CONDENSED_OFC2 = read_tensor<int64_t>(condense_path + "/CONDENSED_DEMUX_OFC2.bin");
    tensor2array<int64_t>(CONDENSED_OFC2, REF_CONDENSED_OFC2, 1, 1, 1, 1, 2*NUM_X, 2*NUM_X);

    //* 读取各阶段参考值
    auto MHA_X          = read_tensor<int64_t>(binaries_path + "/MHA_LN_X.bin"    );
    auto MHA_O          = read_tensor<int64_t>(binaries_path + "/MHA_O.bin"       );
    auto MHA_O_RES      = read_tensor<int64_t>(binaries_path + "/MHA_O_RES.bin"   );
    auto MLP_X          = read_tensor<int64_t>(binaries_path + "/MLP_LN_X.bin"    );
    auto MLP_XFC2       = read_tensor<int64_t>(binaries_path + "/MLP_XFC2.bin"    );
    auto MLP_XFC2_RES   = read_tensor<int64_t>(binaries_path + "/MLP_XFC2_RES.bin");

    //             dtype,   tensor,      array,            H_LOAD, H, T_LOAD, T_START, T,  C_LOAD, C
    tensor2array<int64_t>(MHA_X,        REF_MHA_X,        1, 1,  T_LOAD, POS, T,  C,  C);
    tensor2array<int64_t>(MHA_O,        REF_MHA_O,        1, 1,  T_LOAD, POS, T,  C,  C);
    tensor2array<int64_t>(MLP_X,        REF_MLP_X,        1, 1,  T_LOAD, POS, T,  C,  C);
    tensor2array<int64_t>(MLP_XFC2,     REF_MLP_FC2,      1, 1,  T_LOAD, POS, T,  C,  C);
    tensor2array<int64_t>(MHA_O_RES,    REF_MHA_O_RES,    1, 1,  T_LOAD, POS, T,  C,  C);
    tensor2array<int64_t>(MLP_XFC2_RES, REF_MLP_FC2_RES,  1, 1,  T_LOAD, POS, T,  C,  C);

    //* 检验输入自洽性
    for(int t=0; t<T; ++t){
        for(int c=0; c<C; ++c){
            assert(REF_MHA_X    [t*C + c] + REF_MHA_O  [t*C + c] == REF_MHA_O_RES  [t*C + c]);
            assert(REF_MHA_O_RES[t*C + c]                         == REF_MLP_X      [t*C + c]);
            assert(REF_MLP_X    [t*C + c] + REF_MLP_FC2[t*C + c] == REF_MLP_FC2_RES[t*C + c]);
        }
    }

    //* 组装 REF_RESIDUAL_O：pass0=MHA_X, pass1=MLP_X
    for(int pass=0; pass<2; ++pass){
        for(int t=0; t<T; ++t){
            for(int c=0; c<C; ++c){
                int64_t val = (pass == 0) ? REF_MHA_X[t*C + c] : REF_MLP_X[t*C + c];
                REF_RESIDUAL_O[pass*T*C + t*C + c] = val;
            }
        }
    }

    //* REF_Y = MLP_XFC2_RES（最后一层输出）
    for(int t=0; t<T; ++t){
        for(int c=0; c<C; ++c){
            REF_X[t*C + c] = REF_MLP_FC2_RES[t*C + c];  // 复用 REF_X 存 Y ref
        }
    }

    //* 将 OFC2 写入 res_i_stream（CP=8 wide，flat 顺序）
    array2stream<int64_t, VIT_X_T, 1, 1, 1, 1, 2*NUM_X, CP>(REF_CONDENSED_OFC2, res_i_stream, "CONDENSED OFC2", true);

    //* save condensed input
    save_condensed_tensor<int64_t, VIT_X_T, 2*NUM_X, CP>(condense_path + "/CONDENSED_RESIDUAL_I.bin", res_i_stream);

    //* 仅第一层：将初始 x 写入 x_stream
    if(l == l_begin){
        // 初始 x = 第一层的 MHA_LN_X（MHA 之前的 x）
        int64_t init_x[T*C];
        auto init_x_tensor = read_tensor<int64_t>(binaries_path + "/MHA_LN_X.bin");
        tensor2array<int64_t>(init_x_tensor, init_x, 1, 1, T_LOAD, POS, T, C, C);
        array2stream<int64_t, VIT_X_T, 1, 1, T, TP, C, CP>(init_x, x_stream, "X", true);
        save_condensed_tensor<int64_t, VIT_X_T, T*C, TP*CP>(condense_path + "/CONDENSED_RESIDUAL_X.bin", x_stream);
    }
}


void compare_encoder_data(
    int l,
    int l_begin,
    int l_close,
    const string& binaries_path,
    const string& condense_path,
    hls::stream<hls::vector<VIT_X_T, TP*CP> >& res_o_stream,
    hls::stream<hls::vector<VIT_X_T, TP*CP> >& y_stream
){
    //* 保存并校验 res_o（共 2*T*C 个元素：pass0 MHA_X + pass1 MLP_X）
    hls::stream<hls::vector<VIT_X_T, TP*CP> > sim_res_o_stream("sim_res_o_stream");
    stream2stream                    <VIT_X_T, 2*T*C, TP*CP>(res_o_stream, sim_res_o_stream);
    save_condensed_tensor<int64_t, VIT_X_T, 2*T*C, TP*CP>(condense_path + "/CONDENSED_RESIDUAL_O.bin", sim_res_o_stream);
    stream2array<int64_t, VIT_X_T, 2, T, TP, C, CP>(sim_res_o_stream, DUT_RESIDUAL_O, "RESIDUAL O", true);
    assert(sim_res_o_stream.size() == 0);
    compare<int64_t>(REF_RESIDUAL_O, DUT_RESIDUAL_O, 2*T*C, "RESIDUAL");

    //* 最后一层：校验 y（MLP_XFC2_RES）
    if(l == l_close-1){
        hls::stream<hls::vector<VIT_X_T, TP*CP> > sim_y_stream("sim_y_stream");
        stream2stream                    <VIT_X_T, T*C, TP*CP>(y_stream, sim_y_stream);
        save_condensed_tensor<int64_t, VIT_X_T, T*C, TP*CP>(condense_path + "/CONDENSED_RESIDUAL_Y.bin", sim_y_stream);
        stream2array<int64_t, VIT_X_T, 1, T, TP, C, CP>(sim_y_stream, DUT_Y, "Y", true);
        compare<int64_t>(REF_X, DUT_Y, T*C, "Y");   // REF_X 存的是 MLP_XFC2_RES
    }
}


void test_layer(int l_begin, int l_close){
    //* create streams
    hls::stream<hls::vector<VIT_X_T, TP*CP> > x_stream      ("X"         );
    hls::stream<hls::vector<VIT_X_T,    CP> > res_i_stream   ("RESIDUAL I");
    hls::stream<hls::vector<VIT_X_T, TP*CP> > res_o_stream   ("RESIDUAL O");
    hls::stream<hls::vector<VIT_X_T, TP*CP> > y_stream       ("Y"         );

    //* for each layer, prepare the data
    for(int l=l_begin; l<l_close && l<VIT_L; ++l){
        prepare_encoder_data(
            l,
            l_begin,
            l_close,
            VIT_BINARIES_PATH + to_string(l),
            VIT_CONDENSE_PATH + to_string(l),
            x_stream,
            res_i_stream
        );
    }

    //* call top function
    top(l_begin, l_close, x_stream, res_i_stream, res_o_stream, y_stream);

    //* for each layer, compare the data
    for(int l=l_begin; l<l_close && l<VIT_L; ++l){
        compare_encoder_data(
            l,
            l_begin,
            l_close,
            VIT_BINARIES_PATH + to_string(l),
            VIT_CONDENSE_PATH + to_string(l),
            res_o_stream,
            y_stream
        );
    }
}


int main(){
    for(int l=0; l<1; ++l){
        test_layer(l, l+1);
        printf("Layer %d passed\n", l);
    }

    return 0;
}
