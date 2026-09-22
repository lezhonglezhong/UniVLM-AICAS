# 方法说明

## System partition

The platform combines a KV260 ARM processing system (PS), programmable logic (PL), and external DDR.

1. The PS prepares multimodal inputs with the SmolVLM2 processor and tokenizer, expands image splits, and produces patch/position embeddings.
2. Each real image split is scheduled through the PL vision encoder in processor order. The historical path uses a 12-layer ViT with W5A8 packed parameters.
3. The PS applies the ViT final post-layer normalization and Connector, then merges visual features into the text embedding sequence.
4. The PL executes the 32-layer LLM prefill and autoregressive decode. External DDR holds packed weights, residual state, and the LLM KV cache.
5. A PYNQ host driver controls the overlay, AXI-Lite registers, DDR buffers, mode/run masks, and result collection.

## Measurement protocol

| Evaluation | Protocol summary |
| --- | --- |
| Accuracy | OCRBench stratified `sample-30`; report correct answers over 30 cases. |
| Throughput | Fixed long prompt and image; report prefill and decode separately, including prompt/completion tokens and the active clock policy. |
| Energy | Read the board PMBus/sysfs power sensor at 100 Hz over the inference window; report completion tokens, energy, average power, and tokens/J. |
| TTFT | Use image/prompt combinations with a cache-busting nonce; fit time to first token against prompt length per image and aggregate the linear slope/intercept. |

## Interpretation guardrails

- Dynamic frequency was used only in the archived throughput run (prepare/prefill at 300 MHz and decode at 250 MHz). Accuracy, energy, and TTFT use static 250 MHz as stated in the result table.
- The score card does not claim end-to-end throughput for arbitrary prompts or images; it is a defined benchmark case.
- The 15/30 OCRBench value is a sample-30 result, not a full-dataset accuracy claim.
- Weight/overlay compatibility is a functional requirement, not merely a performance setting.
