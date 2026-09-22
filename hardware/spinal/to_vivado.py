from pathlib import Path
import re
import shutil


# 复用版 Vivado 导出脚本：
# 1. 输入来自 Spinal 生成的 ACCELERATOR.v / ACCELERATOR_bb.v。
# 2. 将 HLS all.v 中的 $readmemh 绝对路径改为 Vivado 目录下的相对 dat 文件。
# 3. 收集 src/main/verilog/<复用IP>/*.dat，避免沿用旧 Qwen 模块名。

ROOT = Path(".")
VIVADO_DIR = ROOT / "vivado"
SPINAL_RTL = [ROOT / "ACCELERATOR.v", ROOT / "ACCELERATOR_bb.v"]
READMEM_RE = re.compile(r'\$readmemh\("(.*?)"')


def reset_vivado_dir() -> None:
    VIVADO_DIR.mkdir(exist_ok=True)
    for suffix in ("*.v", "*.dat"):
        for path in VIVADO_DIR.glob(suffix):
            path.unlink()


def rewrite_readmem_paths(src: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(
            f"{src} 不存在；请先在 SPINAL 目录运行 ./tools/sbt \"runMain generate_accelerator\""
        )

    contents = []
    for line in src.read_text().splitlines(keepends=True):
        match = READMEM_RE.search(line)
        if match:
            old_path = match.group(1)
            line = line.replace(old_path, f"./{Path(old_path).name}")
        contents.append(line)

    dst = VIVADO_DIR / src.name.replace(".v", "_replaced.v")
    dst.write_text("".join(contents))


def copy_hls_dat_files() -> None:
    for dat_file in (ROOT / "src" / "main" / "verilog").glob("*/*.dat"):
        shutil.copy2(dat_file, VIVADO_DIR / dat_file.name)


def main() -> None:
    reset_vivado_dir()
    for rtl in SPINAL_RTL:
        rewrite_readmem_paths(rtl)
    copy_hls_dat_files()


if __name__ == "__main__":
    main()
