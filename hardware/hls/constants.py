import os

# current dir
ROOT_DIR = os.getcwd()

# some global settings
# 当前 VLM 仓库内自带 SPINAL/ 工程，Step3 导出和 SpinalHDL 封装都写入本目录。
# 旧 Qwen 流程使用 ../SPINAL，会在当前 workspace 中落到仓库外且不可写。
SPINAL_DIR = os.path.join(ROOT_DIR, "SPINAL")
print(f"ROOT_DIR: {ROOT_DIR}, SPINAL_DIR: {SPINAL_DIR}")

if __name__ == "__main__":
    L               = 32 # 32 decoders in llama2
    G               = 8
    DW_WQ           = 4
    DW_WS           = 4
    BURST_LEN       = 256
    DW_MAXI         = G*G*DW_WQ
    C               = 4096
    CM              = 11008
    NUM_Q           = 4*C*C + 3*C*CM
    NUM_S           = NUM_Q // G
    DECODER_BITS    = NUM_Q*DW_WQ + NUM_S*DW_WS*2
    DECODER_CYCS    = DECODER_BITS // DW_MAXI
    DECODER_BURSTS  = DECODER_CYCS // BURST_LEN
    print(f"DECODER_BITS: {DECODER_BITS}, DECODER_CYCS: {DECODER_CYCS}, DECODER_BURSTS: {DECODER_BURSTS}")

    # for whole network, how many bits?
    LLAMA_BITS      = L * DECODER_BITS
    LLAMA_BYTES     = LLAMA_BITS // 8
    LLAMA_GBs       = LLAMA_BYTES / (1<<30)
    LLAMA_CYCS      = L * DECODER_CYCS
    print(f"LLAMA_BYTES: {LLAMA_BYTES}, LLAMA_GBs: {LLAMA_GBs}, LLAMA_CYCS: {LLAMA_CYCS}")
