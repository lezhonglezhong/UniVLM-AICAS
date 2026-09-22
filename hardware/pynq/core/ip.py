"""
FPGA 硬件寄存器访问驱动 (基于 /dev/mem 内存映射)

复用 Qwen PYNQ 版本的 AXI 驱动，用于 SmolVLM2 加速器。
通过 mmap 映射 /dev/mem，以 numpy buffer 方式读写 FPGA PL 侧的 AXI-Lite 控制寄存器。

核心类/函数：
  - AXI_MEM:      将一段物理地址映射为 numpy 数组，支持按索引读写
  - AXI_REGISTER: AXI_MEM 的别名，用于单个寄存器
  - AXI_IP:       工厂函数，创建属性式访问的 IP 驱动对象

使用条件：
  - 需要 root 权限（访问 /dev/mem）
  - 物理地址由 Vivado Block Design 中 AXI-Lite 地址映射决定
"""
import mmap
import os
import numpy as np
from time import sleep


class AXI_MEM:
    """
    物理内存映射类：将物理地址映射为 numpy 数组。

    参数：
        base_addr (int): 目标物理基地址
        dtype (np.dtype): 数据类型
        length (int):     元素个数（与 bytes 互斥）
        bytes (int):      总字节数（与 length 互斥）
    """

    def __init__(self, base_addr, dtype: np.dtype, length=None, bytes=None):
        if length is None and bytes is None:
            length = 1
        elif length is None and bytes is not None:
            length = bytes // np.dtype(dtype).itemsize
        elif length is not None and bytes is None:
            pass
        else:
            raise ValueError("length and bytes are mutually exclusive")

        self.base_addr = base_addr
        self.dtype = dtype
        self.length = length
        self.size = np.dtype(dtype).itemsize * length

        self.page_offset = base_addr % mmap.PAGESIZE
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mem = mmap.mmap(
            self.fd, self.size + self.page_offset,
            mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
            offset=base_addr - self.page_offset
        )
        self.buffer = np.frombuffer(self.mem, dtype=dtype, count=length, offset=self.page_offset)
        self.closed = False

    def read(self, index=0):
        return self.buffer[index]

    def write(self, value, index=0):
        self.buffer[index] = value

    def close(self):
        """先释放 numpy mmap 视图，再关闭 mmap，避免进程退出时报 BufferError。"""
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "buffer"):
            del self.buffer
        if hasattr(self, "mem"):
            self.mem.close()
        if hasattr(self, "fd"):
            os.close(self.fd)

    def __del__(self):
        try:
            self.close()
        except Exception:
            # 析构期可能处于解释器退出阶段，此处不能再抛异常干扰真实运行结果。
            pass

    def __getitem__(self, index):
        return self.read(index)

    def __setitem__(self, index, value):
        self.write(value, index)


AXI_REGISTER = AXI_MEM


def AXI_IP(base_addr, reg_list):
    """
    IP 驱动工厂函数。

    参数：
        base_addr (int):  IP 的 AXI-Lite 基地址
        reg_list (list):  [(name, offset, dtype), ...] 寄存器描述列表

    返回：
        可用属性方式读写寄存器的对象
    """
    class _AXI_IP:
        def __init__(self, _base_addr, _reg_list):
            self.registers = {}
            for name, offset, dtype in _reg_list:
                abs_addr = _base_addr + offset
                reg = AXI_REGISTER(abs_addr, dtype)
                self.registers[name] = reg
                self._create_property(name)

        def _create_property(self, name):
            def getter(self):
                return self.registers[name].read()
            def setter(self, value):
                self.registers[name].write(value)
            setattr(self.__class__, name, property(getter, setter))

    return _AXI_IP(base_addr, reg_list)
