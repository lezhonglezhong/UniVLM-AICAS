#include "../src/common.h"
#include "../src/utils.h"
#include "../src/buffer.h"

// ============================================================================
// Vision Encoder (ViT) GEMM_MUX
// ============================================================================
// 将各上游模块输出汇聚到主 GEMM 输入流（l < VIT_L）：
//   XLN → QKV; A → O; XLN → FC1; XM → FC2
// 已去掉 CLS (LM Head) 与 Connector 路径
//
// Option A 修正：MHA-QKV 使用全局大 buffer 缓存全部 T=1024 tokens，
// 外层循环改为 [h][qkv]，使维度顺序与 PERMUTE testbench 和 DEMUX 对齐：
//   PERMUTE i_stream: [h][qkv][tt][hct][ct]
//   DEMUX   读取顺序: [h][qkv][tt][hct][t]  ← 均以 [h][qkv] 为外层

// simulation hyperparameters
constexpr int L         = VIT_L;       // 12
constexpr int T_LOAD    = 5120;

// model hyperparameters
constexpr int H         = VIT_H;       // 12
constexpr int G         = VIT_G;       // 8
constexpr int C         = VIT_C;       // 768
constexpr int HC        = VIT_HC;      // 64
constexpr int CM        = VIT_MLP_DIM; // 3072

// design hyperparameters
constexpr int T         = VIT_S;  // 1024 (total tokens)
constexpr int TP_TILE   = 8;      // GEMM 硬件 token 并行度（输出流宽度）
constexpr int TP        = 1;      // 上游输入流 token 粒度（LayerNorm 每拍 1 token）
constexpr int CP        = G;      // channel 并行度 = 8

// derived hyperparameters
constexpr int TT        = T  / TP_TILE; // 128  token tiles
constexpr int CT        = C  / CP;      // 96   channel tiles
constexpr int CMT       = CM / CP;      // 384  MLP channel tiles
constexpr int HCT       = HC / CP;      // 8    head-channel tiles
typedef ap_int <DW_AQ> aq_t;
typedef ap_uint<DW_AS> as_t;
typedef ap_uint<DW_AQ * CP>      aq_pack_t;
typedef ap_uint<DW_AS * TP_TILE> as_pack_t;

// per-tile output column group counts (unchanged semantics)
constexpr int R_QKV     = H * 3 * HCT; // 288  (QKV 每 tile 输出列组数)
constexpr int R_O       = H * 1 * HCT; // 96   (O   每 tile 输出列组数)
constexpr int R_M       = 1 * CMT;     // 384  (FC1 每 tile 输出列组数)
constexpr int R_D       = 1 * CT;      // 96   (FC2 每 tile 输出列组数)

// Total activation elements over all T tokens
constexpr int NUM_X     = (3*CT*T*C + CT*T*C + CMT*T*C + CT*T*CM);
constexpr int NUM_WS    = NUM_X / G;

// ============================================================================
// 大 buffer：缓存 MHA 阶段全部 T=1024 tokens 的 XLN 激活。
// Q 以 [TP_TILE][TT][CT] 存 8 个 64-bit bank，S 以 [TT][CT] 存 32-bit pack。
// 每个物理 bank 宽度 <=72 bit，适配 URAM 宽度，读出时仍还原成 TP_TILE*CP=64 元素。
//
// 为什么需要大 buffer：
//   QKV 外层循环为 [h][qkv]（共 H*3=36 次），每次需要访问全部 TT=128 个
//   token tile；若仅缓存 1 tile，则每次外层迭代都需重新从流读取，
//   但流只能顺序读一次。因此必须先将所有 T tokens 读入 buffer，再重放。
//
// 大小：128 × 96 × 8 × 8 = 786,432 int8 ≈ 768 KB
// ============================================================================
aq_pack_t xln_mha_buf_q[TP_TILE][TT][CT];
as_pack_t xln_mha_buf_s[TT][CT];

static ap_uint<DW_AQ> aq_to_bits(aq_t val){
    #pragma HLS inline
    ap_uint<DW_AQ> bits = val.range(DW_AQ - 1, 0);
    return bits;
}

static aq_t aq_from_bits(ap_uint<DW_AQ> bits){
    #pragma HLS inline
    aq_t val;
    val.range(DW_AQ - 1, 0) = bits;
    return val;
}

static aq_pack_t pack_aq_cp(const hls::vector<aq_t, CP>& vec){
    #pragma HLS inline
    aq_pack_t pack = 0;
    pack.range(1 * DW_AQ - 1, 0 * DW_AQ) = aq_to_bits(vec[0]);
    pack.range(2 * DW_AQ - 1, 1 * DW_AQ) = aq_to_bits(vec[1]);
    pack.range(3 * DW_AQ - 1, 2 * DW_AQ) = aq_to_bits(vec[2]);
    pack.range(4 * DW_AQ - 1, 3 * DW_AQ) = aq_to_bits(vec[3]);
    pack.range(5 * DW_AQ - 1, 4 * DW_AQ) = aq_to_bits(vec[4]);
    pack.range(6 * DW_AQ - 1, 5 * DW_AQ) = aq_to_bits(vec[5]);
    pack.range(7 * DW_AQ - 1, 6 * DW_AQ) = aq_to_bits(vec[6]);
    pack.range(8 * DW_AQ - 1, 7 * DW_AQ) = aq_to_bits(vec[7]);
    return pack;
}

static void unpack_aq_cp(aq_pack_t pack, hls::vector<aq_t, CP>& vec){
    #pragma HLS inline
    vec[0] = aq_from_bits(pack.range(1 * DW_AQ - 1, 0 * DW_AQ));
    vec[1] = aq_from_bits(pack.range(2 * DW_AQ - 1, 1 * DW_AQ));
    vec[2] = aq_from_bits(pack.range(3 * DW_AQ - 1, 2 * DW_AQ));
    vec[3] = aq_from_bits(pack.range(4 * DW_AQ - 1, 3 * DW_AQ));
    vec[4] = aq_from_bits(pack.range(5 * DW_AQ - 1, 4 * DW_AQ));
    vec[5] = aq_from_bits(pack.range(6 * DW_AQ - 1, 5 * DW_AQ));
    vec[6] = aq_from_bits(pack.range(7 * DW_AQ - 1, 6 * DW_AQ));
    vec[7] = aq_from_bits(pack.range(8 * DW_AQ - 1, 7 * DW_AQ));
}

static void set_as_lane(as_pack_t& pack, int tp, as_t val){
    #pragma HLS inline
    switch(tp){
        case 0: pack.range(1 * DW_AS - 1, 0 * DW_AS) = val; break;
        case 1: pack.range(2 * DW_AS - 1, 1 * DW_AS) = val; break;
        case 2: pack.range(3 * DW_AS - 1, 2 * DW_AS) = val; break;
        case 3: pack.range(4 * DW_AS - 1, 3 * DW_AS) = val; break;
        case 4: pack.range(5 * DW_AS - 1, 4 * DW_AS) = val; break;
        case 5: pack.range(6 * DW_AS - 1, 5 * DW_AS) = val; break;
        case 6: pack.range(7 * DW_AS - 1, 6 * DW_AS) = val; break;
        default: pack.range(8 * DW_AS - 1, 7 * DW_AS) = val; break;
    }
}

static as_t get_as_lane(as_pack_t pack, int tp){
    #pragma HLS inline
    switch(tp){
        case 0: return pack.range(1 * DW_AS - 1, 0 * DW_AS);
        case 1: return pack.range(2 * DW_AS - 1, 1 * DW_AS);
        case 2: return pack.range(3 * DW_AS - 1, 2 * DW_AS);
        case 3: return pack.range(4 * DW_AS - 1, 3 * DW_AS);
        case 4: return pack.range(5 * DW_AS - 1, 4 * DW_AS);
        case 5: return pack.range(6 * DW_AS - 1, 5 * DW_AS);
        case 6: return pack.range(7 * DW_AS - 1, 6 * DW_AS);
        default: return pack.range(8 * DW_AS - 1, 7 * DW_AS);
    }
}

// ============================================================================
// 小 buffer：O / FC1 / FC2 按 tile 流式处理（无需大 buffer）
// ============================================================================
BUFFER<aq_t, H, TP_TILE, TP, HC,  CP> a_buffer_q;   // O: 合并 H 个 head
BUFFER<as_t, H, TP_TILE, TP, HCT, 1 > a_buffer_s;

BUFFER<aq_t, 1, TP_TILE, TP, C,   CP> fc1_xln_buf_q; // FC1: XLN 每 tile 重放 R_M 次
BUFFER<as_t, 1, TP_TILE, TP, CT,  1 > fc1_xln_buf_s;

BUFFER<aq_t, 1, TP_TILE, TP, CM,  CP> xm_buf_q;      // FC2: XM (GELU 输出)
BUFFER<as_t, 1, TP_TILE, TP, CMT, 1 > xm_buf_s;

// ============================================================================
// MHA_O_MUX — O 投影每 tile 的 dataflow 辅助函数
// ============================================================================
void MHA_O_MUX(
    hls::stream<hls::vector<aq_t, TP_TILE*CP> > &aq_buf,
    hls::stream<hls::vector<as_t, TP_TILE   > > &as_buf,
    hls::stream<hls::vector<aq_t, TP_TILE*CP> > &q_stream,
    hls::stream<hls::vector<as_t, TP_TILE   > > &s_stream
){
    // R_O*CT = 96*96 = 9216 次，每次转发一个 TP_TILE*CP 宽向量
    for(int n = 0; n < R_O*CT; ++n){
        #pragma HLS pipeline II=1
        q_stream.write(aq_buf.read());
        s_stream.write(as_buf.read());
    }
}

// ============================================================================
// FC1/FC2 MUX — 分开处理，保证全局顺序与 ViT_PERMUTE 一致：
//   先输出所有 tt 的 FC1，再输出所有 tt 的 FC2（不能按 tt 交错 FC1/FC2）
// ============================================================================
void FC1_MUX(
    hls::stream<hls::vector<aq_t, TP_TILE*CP> > &xlnq_buf,
    hls::stream<hls::vector<as_t, TP_TILE   > > &xlns_buf,
    hls::stream<hls::vector<aq_t, TP_TILE*CP> > &q_stream,
    hls::stream<hls::vector<as_t, TP_TILE   > > &s_stream
){
    for(int n = 0; n < R_M*CT; ++n){
        #pragma HLS pipeline II=1
        q_stream.write(xlnq_buf.read());
        s_stream.write(xlns_buf.read());
    }
}

void FC2_MUX(
    hls::stream<hls::vector<aq_t, TP_TILE*CP> > &xmq_buf,
    hls::stream<hls::vector<as_t, TP_TILE   > > &xms_buf,
    hls::stream<hls::vector<aq_t, TP_TILE*CP> > &q_stream,
    hls::stream<hls::vector<as_t, TP_TILE   > > &s_stream
){
    for(int n = 0; n < R_D*CMT; ++n){
        #pragma HLS pipeline II=1
        q_stream.write(xmq_buf.read());
        s_stream.write(xms_buf.read());
    }
}

// ============================================================================
// top
// ============================================================================
void top(
    int l_begin,
    int l_close,
    // 上游输入流（每拍 TP=1 token，宽度 TP*CP=8）
    hls::stream<hls::vector<aq_t, TP*CP    > > &xlnq_stream,
    hls::stream<hls::vector<as_t, TP       > > &xlns_stream,
    hls::stream<hls::vector<aq_t, TP*CP    > > &aq_stream,
    hls::stream<hls::vector<as_t, TP       > > &as_stream,
    hls::stream<hls::vector<aq_t,    CP    > > &xmq_stream,
    hls::stream<hls::vector<as_t,    1     > > &xms_stream,
    // 下游输出流（每拍 TP_TILE=8 tokens，宽度 TP_TILE*CP=64）
    hls::stream<hls::vector<aq_t, TP_TILE*CP> > &q_stream,
    hls::stream<hls::vector<as_t, TP_TILE  > > &s_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=xlnq_stream
    #pragma HLS interface axis port=xlns_stream
    #pragma HLS interface axis port=aq_stream
    #pragma HLS interface axis port=as_stream
    #pragma HLS interface axis port=xmq_stream
    #pragma HLS interface axis port=xms_stream
    #pragma HLS interface axis port=q_stream
    #pragma HLS interface axis port=s_stream
    #pragma HLS aggregate variable=xlnq_stream  compact=bit
    #pragma HLS aggregate variable=xlns_stream  compact=bit
    #pragma HLS aggregate variable=aq_stream    compact=bit
    #pragma HLS aggregate variable=as_stream    compact=bit
    #pragma HLS aggregate variable=xmq_stream   compact=bit
    #pragma HLS aggregate variable=xms_stream   compact=bit
    #pragma HLS aggregate variable=q_stream     compact=bit
    #pragma HLS aggregate variable=s_stream     compact=bit

    #pragma HLS array_partition variable=xln_mha_buf_q dim=1 complete   // TP_TILE=8
    #pragma HLS bind_storage variable=xln_mha_buf_q type=RAM_2P impl=URAM
    #pragma HLS bind_storage variable=xln_mha_buf_s type=RAM_2P impl=URAM

    as_pack_t xln_mha_s_tile[CT];
    #pragma HLS bind_storage variable=xln_mha_s_tile type=ram_2p impl=lutram

    for(int l = l_begin; l < l_close; ++l){
        if(l >= VIT_L) continue;

        // ══════════════════════════════════════════════════════════════
        // MHA Phase
        // ══════════════════════════════════════════════════════════════

        // ── Step 1: 将全部 T=1024 MHA-XLN tokens 读入大 buffer ────────
        // 流顺序：[T][CT] = [TT*TP_TILE][CT]，即 token 在外、channel 在内
        // 循环：[tt][tp][ct]，每拍读入 TP*CP=8 个激活值
        // 存入：Q buf[tp][tt][ct]，S 先在当前 tt 内打包后写入 buf[tt][ct]
        for(int tt = 0; tt < TT; ++tt){
            for(int tp = 0; tp < TP_TILE; ++tp){  // TP_TILE/TP=8 次读取填满一个 tile
                for(int ct = 0; ct < CT; ++ct){
                    #pragma HLS pipeline II=1
                    hls::vector<aq_t, TP*CP> qv = xlnq_stream.read();
                    hls::vector<as_t, TP   > sv = xlns_stream.read();
                    as_pack_t s_pack = 0;
                    if(tp != 0) s_pack = xln_mha_s_tile[ct];
                    xln_mha_buf_q[tp][tt][ct] = pack_aq_cp(qv);
                    set_as_lane(s_pack, tp, sv[0]);
                    xln_mha_s_tile[ct] = s_pack;
                }
            }
            for(int ct = 0; ct < CT; ++ct){
                #pragma HLS pipeline II=1
                xln_mha_buf_s[tt][ct] = xln_mha_s_tile[ct];
            }
        }

        // ── Step 2: QKV GEMM，顺序 [h][qkv][tt][hct][ct] ─────────────
        // 与 PERMUTE testbench i_stream 填充顺序完全一致。
        // hct 循环在激活数据上是 (void)（激活不依赖 hct），仅用于与权重流配对。
        // 每拍输出 TP_TILE*CP=64 个激活值（8 tokens × 8 channels）。
        for(int h = 0; h < H; ++h){
            for(int qkv = 0; qkv < 3; ++qkv){
                for(int tt = 0; tt < TT; ++tt){
                    for(int hct = 0; hct < HCT; ++hct){
                        for(int ct = 0; ct < CT; ++ct){
                            #pragma HLS pipeline II=1
                            hls::vector<aq_t, TP_TILE*CP> q_vec;
                            hls::vector<as_t, TP_TILE   > s_vec;
                            as_pack_t s_pack = xln_mha_buf_s[tt][ct];
                            for(int tp = 0; tp < TP_TILE; ++tp){
                                #pragma HLS unroll
                                hls::vector<aq_t, CP> q_lane;
                                unpack_aq_cp(xln_mha_buf_q[tp][tt][ct], q_lane);
                                for(int cp = 0; cp < CP; ++cp){
                                    #pragma HLS unroll
                                    q_vec[tp*CP + cp] = q_lane[cp];
                                }
                                s_vec[tp] = get_as_lane(s_pack, tp);
                            }
                            q_stream.write(q_vec);
                            s_stream.write(s_vec);
                        }
                    }
                }
            }
        }

        // ── Step 3: O GEMM，顺序 [tt][ct_out][ct]（tile 流式）──────────
        // A tokens 来自注意力模块，TP=1 token/拍，H 个 head 汇聚为 C=768 宽。
        // 与 PERMUTE testbench 及 DEMUX 期望的 [tt]-outer O 顺序一致。
        for(int tt = 0; tt < TT; ++tt){
            #pragma HLS dataflow
            hls::stream<hls::vector<aq_t, TP_TILE*CP> > aq_buf("aq_buf");
            hls::stream<hls::vector<as_t, TP_TILE   > > as_buf("as_buf");
            a_buffer_q.do_buffer_merge(true, R_O, aq_stream, aq_buf);
            a_buffer_s.do_buffer_merge(true, R_O, as_stream, as_buf);
            MHA_O_MUX(aq_buf, as_buf, q_stream, s_stream);
        }

        // ══════════════════════════════════════════════════════════════
        // MLP Phase
        // 全局顺序必须是：FC1(all tt) -> FC2(all tt)
        // ══════════════════════════════════════════════════════════════
        for(int tt = 0; tt < TT; ++tt){
            #pragma HLS dataflow
            hls::stream<hls::vector<aq_t, TP_TILE*CP> > xlnq_buf("xlnq_buf");
            hls::stream<hls::vector<as_t, TP_TILE   > > xlns_buf("xlns_buf");
            fc1_xln_buf_q.do_buffer(true, R_M, xlnq_stream, xlnq_buf);
            fc1_xln_buf_s.do_buffer(true, R_M, xlns_stream, xlns_buf);
            FC1_MUX(xlnq_buf, xlns_buf, q_stream, s_stream);
        }

        for(int tt = 0; tt < TT; ++tt){
            #pragma HLS dataflow
            hls::stream<hls::vector<aq_t, TP_TILE*CP> > xmq_buf ("xmq_buf" );
            hls::stream<hls::vector<as_t, TP_TILE   > > xms_buf ("xms_buf" );
            xm_buf_q.do_buffer_unpack(true, R_D, xmq_stream, xmq_buf);
            xm_buf_s.do_buffer_unpack(true, R_D, xms_stream, xms_buf);
            FC2_MUX(xmq_buf, xms_buf, q_stream, s_stream);
        }
    }
}


// ============================================================================
// Reference data
// ============================================================================

int8_t REF_MHA_XLN_Q   [T  *C  ];
int8_t REF_MHA_XLN_S   [T  *CT ];
int8_t REF_MLP_XLN_Q   [T  *C  ];
int8_t REF_MLP_XLN_S   [T  *CT ];
int8_t REF_XLN_Q        [T*2*C  ];   // concat of MHA+MLP layernorm outputs
int8_t REF_XLN_S        [T*2*CT ];

int8_t REF_MHA_A_Q      [T  *C  ];
int8_t REF_MHA_A_S      [T  *CT ];

int8_t REF_MLP_XM_Q     [T  *CM ];   // GELU output (input to FC2)
int8_t REF_MLP_XM_S     [T  *CMT];

// GEMM condensed reference output
int8_t REF_X_Q          [NUM_X  ];
int8_t REF_X_S          [NUM_WS ];
int8_t DUT_X_Q          [NUM_X  ];
int8_t DUT_X_S          [NUM_WS ];

// ============================================================================
// test_layer — encoder layer
// ============================================================================
void test_layer(int l){
    if(l >= VIT_L) return;

    const string binaries_path = VIT_BINARIES_PATH + to_string(l);
    const string condense_path = VIT_CONDENSE_PATH + to_string(l);
    {
        auto CONDENSED_XLN_Q = read_tensor<int8_t>(condense_path + "/CONDENSED_XLN_Q.bin");
        auto CONDENSED_XLN_S = read_tensor<int8_t>(condense_path + "/CONDENSED_XLN_S.bin");
        auto CONDENSED_A_Q   = read_tensor<int8_t>(condense_path + "/CONDENSED_RV_GEMM_A_Q.bin");
        auto CONDENSED_A_S   = read_tensor<int8_t>(condense_path + "/CONDENSED_RV_GEMM_A_S.bin");
        auto CONDENSED_XM_Q  = read_tensor<int8_t>(condense_path + "/CONDENSED_GELU_XM_Q.bin");
        auto CONDENSED_XM_S  = read_tensor<int8_t>(condense_path + "/CONDENSED_GELU_XM_S.bin");

        tensor2array<int8_t>(CONDENSED_XLN_Q, REF_XLN_Q,    1, 1, 1, 1, 2*T*C,   2*T*C  );
        tensor2array<int8_t>(CONDENSED_XLN_S, REF_XLN_S,    1, 1, 1, 1, 2*T*CT,  2*T*CT );
        tensor2array<int8_t>(CONDENSED_A_Q,   REF_MHA_A_Q,  1, 1, 1, 1,   T*C,     T*C  );
        tensor2array<int8_t>(CONDENSED_A_S,   REF_MHA_A_S,  1, 1, 1, 1,   T*CT,    T*CT );
        tensor2array<int8_t>(CONDENSED_XM_Q,  REF_MLP_XM_Q, 1, 1, 1, 1,   T*CM,    T*CM );
        tensor2array<int8_t>(CONDENSED_XM_S,  REF_MLP_XM_S, 1, 1, 1, 1,   T*CMT,   T*CMT);

        auto X_Q = read_tensor<int8_t>(condense_path + "/CONDENSED_GEMM_X_Q.bin");
        auto X_S = read_tensor<int8_t>(condense_path + "/CONDENSED_GEMM_X_S.bin");
        tensor2array<int8_t>(X_Q, REF_X_Q, 1, 1, 1, 1, NUM_X,  NUM_X );
        tensor2array<int8_t>(X_S, REF_X_S, 1, 1, 1, 1, NUM_WS, NUM_WS);
    }

    hls::stream<hls::vector<aq_t, TP*CP      > > xlnq_stream  ("xlnq_stream"  );
    hls::stream<hls::vector<as_t, TP         > > xlns_stream  ("xlns_stream"  );
    hls::stream<hls::vector<aq_t, TP*CP      > > aq_stream    ("aq_stream"    );
    hls::stream<hls::vector<as_t, TP         > > as_stream    ("as_stream"    );
    hls::stream<hls::vector<aq_t,    CP      > > xmq_stream   ("xmq_stream"   );
    hls::stream<hls::vector<as_t,    1       > > xms_stream   ("xms_stream"   );
    hls::stream<hls::vector<aq_t, TP_TILE*CP > > q_stream     ("q_stream"     );
    hls::stream<hls::vector<as_t, TP_TILE    > > s_stream     ("s_stream"     );

    // T=1024 total tokens: array2stream sends 2*T tokens for MHA+MLP XLN
    array2stream<int8_t, aq_t, 1, 1, 1, 1, 2*T*C,   TP*CP>(REF_XLN_Q,    xlnq_stream, "XLN Q", true);
    array2stream<int8_t, as_t, 1, 1, 1, 1, 2*T*CT,  TP   >(REF_XLN_S,    xlns_stream, "XLN S", true);
    // A 来自 RV_GEMM condensed，内存按 (H, T, HC/HCT) 展开。
    // O 相位每个全局 tt 会调用一次 a_buffer_merge.do_buffer_merge()，该函数一次消费
    // H*TP_TILE*HCT 个向量，且局部读取顺序是 [h][tp][hct]。
    // 因此这里必须按 [tt][h][tp][hct] 喂流，不能直接用 array2stream 的通用顺序。
    for(int tt=0; tt<TT; ++tt){
        for(int h=0; h<H; ++h){
            for(int tp=0; tp<TP_TILE; ++tp){
                for(int hct=0; hct<HCT; ++hct){
                    hls::vector<aq_t, TP*CP> aq_vec;
                    hls::vector<as_t, TP   > as_vec;
                    for(int cp=0; cp<CP; ++cp){
                        #pragma HLS unroll
                        aq_vec[cp] = REF_MHA_A_Q[h*T*HC + (tt*TP_TILE + tp)*HC + hct*CP + cp];
                    }
                    as_vec[0] = REF_MHA_A_S[h*T*HCT + (tt*TP_TILE + tp)*HCT + hct];
                    aq_stream.write(aq_vec);
                    as_stream.write(as_vec);
                }
            }
        }
    }
    array2stream<int8_t, aq_t, 1, 1, 1, 1,   T*CM,     CP>(REF_MLP_XM_Q, xmq_stream,  "XM Q",  true);
    array2stream<int8_t, as_t, 1, 1, 1, 1,   T*CMT,    1 >(REF_MLP_XM_S, xms_stream,  "XM S",  true);

    top(l, l+1, xlnq_stream, xlns_stream, aq_stream, as_stream, xmq_stream, xms_stream,
        q_stream, s_stream);

    // hls::stream::size() 为向量写次数；每拍宽度 TP_TILE*CP
    printf("q_stream size: %d (expect %d)\n", (int)q_stream.size(), NUM_X / (TP_TILE * CP));
    stream2array<int8_t, aq_t, 1, 1, 1, NUM_X,  TP_TILE*CP>(q_stream, DUT_X_Q, "X Q", true);
    stream2array<int8_t, as_t, 1, 1, 1, NUM_WS, TP_TILE   >(s_stream, DUT_X_S, "X S", true);

    constexpr int MAX_PRINT_MISMATCH = 100;
    int mismatch_q = 0;
    int printed_q = 0;
    for(int i=0; i<NUM_X; ++i){
        if(REF_X_Q[i] != DUT_X_Q[i]){
            if(printed_q < MAX_PRINT_MISMATCH){
                printf("X Q mismatch at [%9d]: %5d vs %5d\n", i, REF_X_Q[i], DUT_X_Q[i]);
                printed_q++;
            }
            mismatch_q++;
        }
    }
    printf("X Q mismatch total: %d\n", mismatch_q);

    int mismatch_s = 0;
    int printed_s = 0;
    for(int i=0; i<NUM_WS; ++i){
        if(REF_X_S[i] != DUT_X_S[i]){
            if(printed_s < MAX_PRINT_MISMATCH){
                printf("X S mismatch at [%9d]: %5d vs %5d\n", i, REF_X_S[i], DUT_X_S[i]);
                printed_s++;
            }
            mismatch_s++;
        }
    }
    printf("X S mismatch total: %d\n", mismatch_s);
}


int main(){
    test_layer(0);
    printf("Layer 0 passed\n");
    return 0;
}
