#include "../src/common.h"
#include "../src/silu_em_quant.h"
#include "../src/utils.h"

// simulation hyperparameters
constexpr int L         = LLAMA_L;
constexpr int T_LOAD    = 612;          // saved seq length, since T is in the inner loop
constexpr int POS       = 96;           // random position

// model hyperparameters
constexpr int C         = LLAMA_CM;
constexpr int G         = LLAMA_G;

// design hyperparameters
constexpr int T         = 8;
constexpr int CP        = 8;            // interface parallelism
constexpr int SILU_CP   = 1;            // hardware  parallelism of silu

// derived hyperparameters
constexpr int CT        = C / CP;

// other hyperparameters
constexpr int TRUNC     = MLP_TRUNC_MUL;
constexpr int NUM_UG    = 2*T*C;


// instanciate the silu_em_quant inst
SILU_EM_QUANT <XUG_TRUNC_T, SILU_T, XM_T, AQ_T, AS_T, T, C, CP, SILU_CP, TRUNC> silu_em_quant_inst;


// top function
void top(
    // scalar inputs
    int l_begin,
    int l_close,
    // streams
    hls::stream<hls::vector<XUG_TRUNC_T,   CP > > &ug_stream,
    hls::stream<hls::vector<AQ_T,          CP > > &q_stream,
    hls::stream<hls::vector<AS_T,          1  > > &s_stream
){
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=ug_stream
    #pragma HLS interface axis port=q_stream
    #pragma HLS interface axis port=s_stream

    #pragma HLS aggregate variable=ug_stream compact=bit
    #pragma HLS aggregate variable=q_stream  compact=bit
    #pragma HLS aggregate variable=s_stream  compact=bit

    for(int l=l_begin; l<l_close && l<LLAMA_L; ++l){
        silu_em_quant_inst.do_silu_em_quant(ug_stream, q_stream, s_stream);
    }
}

// declare ref data
int64_t REF_XU   [  T*C];
int64_t REF_XG   [  T*C];
int64_t REF_UG   [2*T*C];
int8_t  REF_XM_Q [T*C ];
int8_t  REF_XM_S [T*CT];
// declare dut data
int8_t  DUT_XM_Q [T*C ];
int8_t  DUT_XM_S [T*CT];

void test_layer(int l){
    //* read input data and reference data
    string file_path = BINARIES_PATH + to_string(l);
    string save_path = CONDENSE_PATH + to_string(l);
    {
        // read refs
        auto REF_XU_TENSOR      = read_tensor<int64_t>(file_path + "/MLP_XU.bin");
        auto REF_XG_TENSOR      = read_tensor<int64_t>(file_path + "/MLP_XG.bin");
        auto REF_XM_Q_TENSOR    = read_tensor<int8_t> (file_path + "/MLP_XM_Q.bin");
        auto REF_XM_S_TENSOR    = read_tensor<int8_t> (file_path + "/MLP_XM_S.bin");
        // check tensor size
        assert(REF_XU_TENSOR  .size() == T_LOAD*C );
        assert(REF_XG_TENSOR  .size() == T_LOAD*C );
        assert(REF_XM_Q_TENSOR.size() == T_LOAD*C );
        assert(REF_XM_S_TENSOR.size() == T_LOAD*CT);
        //                      TENSOR,             ARRAY,      H_LOAD, H,      T_LOAD,     T_START,    T,      C_LOAD, C
        tensor2array<int64_t>(  REF_XU_TENSOR,      REF_XU,     1,      1,      T_LOAD,     POS,        T,      C,      C  );
        tensor2array<int64_t>(  REF_XG_TENSOR,      REF_XG,     1,      1,      T_LOAD,     POS,        T,      C,      C  );
        tensor2array<int8_t> (  REF_XM_Q_TENSOR,    REF_XM_Q,   1,      1,      T_LOAD,     POS,        T,      C,      C  );
        tensor2array<int8_t> (  REF_XM_S_TENSOR,    REF_XM_S,   1,      1,      T_LOAD,     POS,        T,      CT,     CT );
    }

    ////* 将 XU 与 XG 交错为 UG 流，顺序需与 silu_em_quant.h stream_split 一致：ct -> ug -> t -> cp
    //for(int ct=0; ct<CT; ++ct){
    //    for(int ug=0; ug<2; ++ug){
    //        for(int t=0; t<T; ++t){
    //            for(int cp=0; cp<CP; ++cp){
    //                int GLOBAL_INDEX = (ct * (2*T) + ug * T + t) * CP + cp;
    //                int LOCAL_INDEX  = t * C + ct * CP + cp;
    //                int64_t raw = (ug == 0 ? REF_XU : REF_XG)[LOCAL_INDEX];
    //                REF_UG     [GLOBAL_INDEX] = raw;
    //            }
    //        }
    //    }
    //}

    //* create streams
    hls::stream<hls::vector<XUG_TRUNC_T, CP > > ug_stream ("ug_stream");
    hls::stream<hls::vector<AQ_T,        CP > > q_stream  ("q_stream" );
    hls::stream<hls::vector<AS_T,        1  > > s_stream  ("s_stream" );


    //* use condensed input
    auto CONDENSED_UG = read_tensor<int64_t>(save_path + "/CONDENSED_DEMUX_UG.bin");
    tensor2array<int64_t>(CONDENSED_UG, REF_UG, 1, 1, 1, 1, NUM_UG, NUM_UG);


    //* write input data to stream
    array2stream<int64_t, XUG_TRUNC_T, 1, 1, 1, 1, NUM_UG, CP>(REF_UG, ug_stream, "CONDENSED UG", true);


    //* call top function
    top(l, l+1, ug_stream, q_stream, s_stream);


    //* save condensed output
    save_condensed_tensor<int8_t, AQ_T, T*C,  CP>(save_path + "/CONDENSED_SILU_EM_QUANT_XM_Q.bin", q_stream);
    save_condensed_tensor<int8_t, AS_T, T*CT, 1 >(save_path + "/CONDENSED_SILU_EM_QUANT_XM_S.bin", s_stream);


    //* read output stream
    stream2array_unpack<int8_t, AQ_T,              1,      T,  T,  C,  CP>(q_stream,   DUT_XM_Q,   "XM_Q", true);
    stream2array_unpack<int8_t, AS_T,              1,      T,  T,  CT, 1 >(s_stream,   DUT_XM_S,   "XM_S", true);


    //* check
    compare<int8_t>(REF_XM_Q, DUT_XM_Q, T*C,  "XM_Q");
    compare<int8_t>(REF_XM_S, DUT_XM_S, T*CT, "XM_S");
}


int main(){
    for(int l=0; l<1; ++l){
        test_layer(l);
        printf("layer %d passed\n", l);
    }

    // test_layer(0);

    return 0;
}