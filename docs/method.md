# 方法说明

UniVLM 在 KV260 的 ARM Processing System（PS）、Programmable Logic（PL）与外部 DDR 间协同执行多模态推理。

1. PS 使用 SmolVLM2 processor 和 tokenizer 准备图像、文本以及 patch/position embedding。
2. PL 执行 12-layer SigLIP ViT 编码；多子图按 processor 顺序串行处理。
3. PS 端完成 ViT post-LayerNorm、Connector 和图像/文本 embedding merge。
4. PL 执行 32-layer LLaMA3 Decoder 的 Prefill 与逐 token Decode；DDR 保存 packed weights、state 和 KV cache。
5. PYNQ 驱动加载 overlay、分配 DDR buffer、配置 AXI-Lite，并收集推理结果与时延。

评测包括 OCRBench accuracy、Prefill/Decode throughput、PMBus power/energy 和 TTFT。
