# PYNQ Core

本目录是当前正式 PYNQ 驱动开发入口。旧 `测试程序_核心代码/` 暂时保留作兼容参考。

当前状态：

- `demo.py` 的 text-only `messages` 已改为 HuggingFace typed content。
- `constants.py` 已改为当前复用硬件寄存器、DDR buffer、RUN_MASK、PL packed 权重和 PS 浮点资产常量。
- `ps_float.py` 已实现文本 embedding、ViT patch/position embedding、ViT final post-LayerNorm、PS Connector、image/text embedding merge，以及 processor 多子图输出 `[B,N_images,3,512,512]` 的展开、padding 过滤和 patch mask 构造。PL/PS state 边界默认使用软件整数模型的 `O_res=12`，即 `scale=2^-12`。
- `fpga_smolvlm2.py` 已改为当前 overlay 的可上板 skeleton：加载 bit/hwh、分配 DDR、加载 PL packed 权重/参数、写入 AXI-Lite 地址寄存器，并提供 LLM/ViT 的 coarse RUN_MASK 调度 helper。
- `fpga_smolvlm2.py` 已新增多子图串行 Vision helper：每个真实子图独立执行 patch/position embedding -> PL ViT 12 层 -> PS final post-LN/Connector，累积为 `[N_real_images,64,960]` 后再 merge 到 `<image>` token 位置。
- `local_inference.py` 是 demo/eval 共享的当前硬件入口；完整自回归生成闭合前，默认只跑 single-chunk smoke 并把 JSON 写到 `../results/`。

本机不能运行 PYNQ overlay；后续板上 bring-up 结果放入 `../results/` 后继续修正。
