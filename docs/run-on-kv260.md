# KV260 运行说明

`hardware/pynq/core/demo.py` 是 PYNQ Demo 入口。部署需要 KV260、PYNQ/XRT 环境、匹配的 overlay、W5A8 packed weights、模型文件和运行时依赖。

```bash
export XILINX_XRT=/usr
python core/demo.py \
  --image <image-path> \
  --prompt "Describe this image." \
  --max-new-tokens 8 \
  --fclk0-mhz 250
```

执行结果会输出生成文本，并写入包含 Prefill、Decode 与总耗时的 JSON 文件。
