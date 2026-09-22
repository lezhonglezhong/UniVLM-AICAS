import spinal.core._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axis.Axi4Stream
import spinal.lib.bus.amba4.axis.Axi4Stream.Axi4Stream
import utils._

import scala.language.postfixOps

// @formatter:off

// 本文件只保留复用版 HLS IP 的黑盒封装。mode=0 为 LLM，mode=1 为 ViT；
// RUN_MASK 由 Controller/PYNQ 选择本次触发的物理 IP，避免旧 Qwen 单链路一次性启动全部模块。

class PERMUTE_Blackbox extends BlackBox {
  val top_name: String = "PERMUTE"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)

    val ap_ctrl:  ApChain      = slave (ApChain())
    val i_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("i_stream",  verilog_file_path)))
    val s_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("s_stream",  verilog_file_path)))
    val w_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("w_stream",  verilog_file_path)))
    val s1_stream:BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("s1_stream", verilog_file_path)))
    val s2_stream:BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("s2_stream", verilog_file_path)))
    val o_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("o_stream",  verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class PERMUTE extends Component {
  setDefinitionName("PERMUTE_wrapper")
  private val black_box = new PERMUTE_Blackbox
  val io = new Bundle {
    val signals:  DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val i_stream: Axi4Stream = slave (Axi4Stream(black_box.io.i_stream .config.to_std_config()))
    val s_stream: Axi4Stream = slave (Axi4Stream(black_box.io.s_stream .config.to_std_config()))
    val w_stream: Axi4Stream = slave (Axi4Stream(black_box.io.w_stream .config.to_std_config()))
    val s1_stream:Axi4Stream = slave (Axi4Stream(black_box.io.s1_stream.config.to_std_config()))
    val s2_stream:Axi4Stream = slave (Axi4Stream(black_box.io.s2_stream.config.to_std_config()))
    val o_stream: Axi4Stream = master(Axi4Stream(black_box.io.o_stream .config.to_std_config()))
    val idle:     Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_PERMUTE)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode    := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin := manager.io.signals.O.L_BEGIN
  black_box.io.l_close := manager.io.signals.O.L_CLOSE

  black_box.io.i_stream .connect2std(io.i_stream)
  black_box.io.s_stream .connect2std(io.s_stream)
  black_box.io.w_stream .connect2std(io.w_stream)
  black_box.io.s1_stream.connect2std(io.s1_stream)
  black_box.io.s2_stream.connect2std(io.s2_stream)
  black_box.io.o_stream .connect2std(io.o_stream)
  Axi4StreamSpecRenamer(io.i_stream)
  Axi4StreamSpecRenamer(io.s_stream)
  Axi4StreamSpecRenamer(io.w_stream)
  Axi4StreamSpecRenamer(io.s1_stream)
  Axi4StreamSpecRenamer(io.s2_stream)
  Axi4StreamSpecRenamer(io.o_stream)
}

class DEMUX_Blackbox extends BlackBox {
  val top_name: String = "DEMUX"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val pos_r:    UInt = in UInt(ctrl_cfg.REG_BITS bits)

    val ap_ctrl:      ApChain      = slave (ApChain())
    val gemm_stream:  BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("gemm_stream",   verilog_file_path)))
    val bias_stream:  BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("bias_stream",   verilog_file_path)))
    val qk_stream:    BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("qk_stream",     verilog_file_path)))
    val v_stream:     BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("v_stream",      verilog_file_path)))
    val mlp1_stream:  BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("mlp1_stream",   verilog_file_path)))
    val od_fc2_stream:BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("od_fc2_stream", verilog_file_path)))
    val cls_stream:   BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("cls_stream",    verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class DEMUX extends Component {
  setDefinitionName("DEMUX_wrapper")
  private val black_box = new DEMUX_Blackbox
  val io = new Bundle {
    val signals:       DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val gemm_stream:   Axi4Stream = slave (Axi4Stream(black_box.io.gemm_stream.config.to_std_config()))
    val bias_stream:   Axi4Stream = slave (Axi4Stream(black_box.io.bias_stream.config.to_std_config()))
    val qk_stream:     Axi4Stream = master(Axi4Stream(black_box.io.qk_stream.config.to_std_config()))
    val v_stream:      Axi4Stream = master(Axi4Stream(black_box.io.v_stream.config.to_std_config()))
    val mlp1_stream:   Axi4Stream = master(Axi4Stream(black_box.io.mlp1_stream.config.to_std_config()))
    val od_fc2_stream: Axi4Stream = master(Axi4Stream(black_box.io.od_fc2_stream.config.to_std_config()))
    val cls_stream:    Axi4Stream = master(Axi4Stream(black_box.io.cls_stream.config.to_std_config()))
    val idle:          Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_DEMUX)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode    := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin := manager.io.signals.O.L_BEGIN
  black_box.io.l_close := manager.io.signals.O.L_CLOSE
  black_box.io.pos_r   := manager.io.signals.O.POS

  black_box.io.gemm_stream  .connect2std(io.gemm_stream)
  black_box.io.bias_stream  .connect2std(io.bias_stream)
  black_box.io.qk_stream    .connect2std(io.qk_stream)
  black_box.io.v_stream     .connect2std(io.v_stream)
  black_box.io.mlp1_stream  .connect2std(io.mlp1_stream)
  black_box.io.od_fc2_stream.connect2std(io.od_fc2_stream)
  black_box.io.cls_stream   .connect2std(io.cls_stream)
  Axi4StreamSpecRenamer(io.gemm_stream)
  Axi4StreamSpecRenamer(io.bias_stream)
  Axi4StreamSpecRenamer(io.qk_stream)
  Axi4StreamSpecRenamer(io.v_stream)
  Axi4StreamSpecRenamer(io.mlp1_stream)
  Axi4StreamSpecRenamer(io.od_fc2_stream)
  Axi4StreamSpecRenamer(io.cls_stream)
}

class ROPE_QK_Blackbox extends BlackBox {
  val top_name: String = "ROPE_QK"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val pos_r:    UInt = in UInt(ctrl_cfg.REG_BITS bits)

    val ap_ctrl:     ApChain      = slave (ApChain())
    val qk_i_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("qk_i_stream", verilog_file_path)))
    val qk_q_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("qk_q_stream", verilog_file_path)))
    val qk_s_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("qk_s_stream", verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class ROPE_QK extends Component {
  setDefinitionName("ROPE_QK_wrapper")
  private val black_box = new ROPE_QK_Blackbox
  val io = new Bundle {
    val signals:     DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val qk_i_stream: Axi4Stream = slave (Axi4Stream(black_box.io.qk_i_stream.config.to_std_config()))
    val qk_q_stream: Axi4Stream = master(Axi4Stream(black_box.io.qk_q_stream.config.to_std_config()))
    val qk_s_stream: Axi4Stream = master(Axi4Stream(black_box.io.qk_s_stream.config.to_std_config()))
    val idle:        Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_ROPE_QK)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode    := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin := manager.io.signals.O.L_BEGIN
  black_box.io.l_close := manager.io.signals.O.L_CLOSE
  black_box.io.pos_r   := manager.io.signals.O.POS

  black_box.io.qk_i_stream.connect2std(io.qk_i_stream)
  black_box.io.qk_q_stream.connect2std(io.qk_q_stream)
  black_box.io.qk_s_stream.connect2std(io.qk_s_stream)
  Axi4StreamSpecRenamer(io.qk_i_stream)
  Axi4StreamSpecRenamer(io.qk_q_stream)
  Axi4StreamSpecRenamer(io.qk_s_stream)
}

class QK_GEMM_Blackbox extends BlackBox {
  val top_name: String = "QK_GEMM"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val pos_r:    UInt = in UInt(ctrl_cfg.REG_BITS bits)

    val ap_ctrl:           ApChain      = slave (ApChain())
    val qk_q_stream:       BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("qk_q_stream",       verilog_file_path)))
    val qk_s_stream:       BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("qk_s_stream",       verilog_file_path)))
    val kq_cache_i_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("kq_cache_i_stream", verilog_file_path)))
    val ks_cache_i_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("ks_cache_i_stream", verilog_file_path)))
    val kq_cache_o_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("kq_cache_o_stream", verilog_file_path)))
    val ks_cache_o_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("ks_cache_o_stream", verilog_file_path)))
    val r_stream:          BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("r_stream",          verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class QK_GEMM extends Component {
  setDefinitionName("QK_GEMM_wrapper")
  private val black_box = new QK_GEMM_Blackbox
  val io = new Bundle {
    val signals:           DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val qk_q_stream:       Axi4Stream = slave (Axi4Stream(black_box.io.qk_q_stream.config.to_std_config()))
    val qk_s_stream:       Axi4Stream = slave (Axi4Stream(black_box.io.qk_s_stream.config.to_std_config()))
    val kq_cache_i_stream: Axi4Stream = slave (Axi4Stream(black_box.io.kq_cache_i_stream.config.to_std_config()))
    val ks_cache_i_stream: Axi4Stream = slave (Axi4Stream(black_box.io.ks_cache_i_stream.config.to_std_config()))
    val kq_cache_o_stream: Axi4Stream = master(Axi4Stream(black_box.io.kq_cache_o_stream.config.to_std_config()))
    val ks_cache_o_stream: Axi4Stream = master(Axi4Stream(black_box.io.ks_cache_o_stream.config.to_std_config()))
    val r_stream:          Axi4Stream = master(Axi4Stream(black_box.io.r_stream.config.to_std_config()))
    val idle:              Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_QK_GEMM)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode    := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin := manager.io.signals.O.L_BEGIN
  black_box.io.l_close := manager.io.signals.O.L_CLOSE
  black_box.io.pos_r   := manager.io.signals.O.POS

  black_box.io.qk_q_stream      .connect2std(io.qk_q_stream)
  black_box.io.qk_s_stream      .connect2std(io.qk_s_stream)
  black_box.io.kq_cache_i_stream.connect2std(io.kq_cache_i_stream)
  black_box.io.ks_cache_i_stream.connect2std(io.ks_cache_i_stream)
  black_box.io.kq_cache_o_stream.connect2std(io.kq_cache_o_stream)
  black_box.io.ks_cache_o_stream.connect2std(io.ks_cache_o_stream)
  black_box.io.r_stream         .connect2std(io.r_stream)
  Axi4StreamSpecRenamer(io.qk_q_stream)
  Axi4StreamSpecRenamer(io.qk_s_stream)
  Axi4StreamSpecRenamer(io.kq_cache_i_stream)
  Axi4StreamSpecRenamer(io.ks_cache_i_stream)
  Axi4StreamSpecRenamer(io.kq_cache_o_stream)
  Axi4StreamSpecRenamer(io.ks_cache_o_stream)
  Axi4StreamSpecRenamer(io.r_stream)
}

class SOFTMAX_Blackbox extends BlackBox {
  val top_name: String = "SOFTMAX"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val pos_r:    UInt = in UInt(ctrl_cfg.REG_BITS bits)

    val ap_ctrl:  ApChain      = slave (ApChain())
    val r_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("r_stream",  verilog_file_path)))
    val rq_stream:BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("rq_stream", verilog_file_path)))
    val rs_stream:BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("rs_stream", verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class SOFTMAX extends Component {
  setDefinitionName("SOFTMAX_wrapper")
  private val black_box = new SOFTMAX_Blackbox
  val io = new Bundle {
    val signals:  DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val r_stream: Axi4Stream = slave (Axi4Stream(black_box.io.r_stream.config.to_std_config()))
    val rq_stream:Axi4Stream = master(Axi4Stream(black_box.io.rq_stream.config.to_std_config()))
    val rs_stream:Axi4Stream = master(Axi4Stream(black_box.io.rs_stream.config.to_std_config()))
    val idle:     Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_SOFTMAX)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode    := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin := manager.io.signals.O.L_BEGIN
  black_box.io.l_close := manager.io.signals.O.L_CLOSE
  black_box.io.pos_r   := manager.io.signals.O.POS

  black_box.io.r_stream .connect2std(io.r_stream)
  black_box.io.rq_stream.connect2std(io.rq_stream)
  black_box.io.rs_stream.connect2std(io.rs_stream)
  Axi4StreamSpecRenamer(io.r_stream)
  Axi4StreamSpecRenamer(io.rq_stream)
  Axi4StreamSpecRenamer(io.rs_stream)
}

class RV_GEMM_Blackbox extends BlackBox {
  val top_name: String = "RV_GEMM"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val pos_r:    UInt = in UInt(ctrl_cfg.REG_BITS bits)

    val ap_ctrl:           ApChain      = slave (ApChain())
    val rq_stream:         BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("rq_stream",         verilog_file_path)))
    val rs_stream:         BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("rs_stream",         verilog_file_path)))
    val v_stream:          BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("v_stream",          verilog_file_path)))
    val vq_cache_i_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("vq_cache_i_stream", verilog_file_path)))
    val vs_cache_i_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("vs_cache_i_stream", verilog_file_path)))
    val vq_cache_o_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("vq_cache_o_stream", verilog_file_path)))
    val vs_cache_o_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("vs_cache_o_stream", verilog_file_path)))
    val aq_stream:         BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("aq_stream",         verilog_file_path)))
    val as_stream:         BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("as_stream",         verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class RV_GEMM extends Component {
  setDefinitionName("RV_GEMM_wrapper")
  private val black_box = new RV_GEMM_Blackbox
  val io = new Bundle {
    val signals:          DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val rq_stream:        Axi4Stream = slave (Axi4Stream(black_box.io.rq_stream.config.to_std_config()))
    val rs_stream:        Axi4Stream = slave (Axi4Stream(black_box.io.rs_stream.config.to_std_config()))
    val v_stream:         Axi4Stream = slave (Axi4Stream(black_box.io.v_stream.config.to_std_config()))
    val vq_cache_i_stream:Axi4Stream = slave (Axi4Stream(black_box.io.vq_cache_i_stream.config.to_std_config()))
    val vs_cache_i_stream:Axi4Stream = slave (Axi4Stream(black_box.io.vs_cache_i_stream.config.to_std_config()))
    val vq_cache_o_stream:Axi4Stream = master(Axi4Stream(black_box.io.vq_cache_o_stream.config.to_std_config()))
    val vs_cache_o_stream:Axi4Stream = master(Axi4Stream(black_box.io.vs_cache_o_stream.config.to_std_config()))
    val aq_stream:        Axi4Stream = master(Axi4Stream(black_box.io.aq_stream.config.to_std_config()))
    val as_stream:        Axi4Stream = master(Axi4Stream(black_box.io.as_stream.config.to_std_config()))
    val idle:             Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_RV_GEMM)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode    := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin := manager.io.signals.O.L_BEGIN
  black_box.io.l_close := manager.io.signals.O.L_CLOSE
  black_box.io.pos_r   := manager.io.signals.O.POS

  black_box.io.rq_stream        .connect2std(io.rq_stream)
  black_box.io.rs_stream        .connect2std(io.rs_stream)
  black_box.io.v_stream         .connect2std(io.v_stream)
  black_box.io.vq_cache_i_stream.connect2std(io.vq_cache_i_stream)
  black_box.io.vs_cache_i_stream.connect2std(io.vs_cache_i_stream)
  black_box.io.vq_cache_o_stream.connect2std(io.vq_cache_o_stream)
  black_box.io.vs_cache_o_stream.connect2std(io.vs_cache_o_stream)
  black_box.io.aq_stream        .connect2std(io.aq_stream)
  black_box.io.as_stream        .connect2std(io.as_stream)
  Axi4StreamSpecRenamer(io.rq_stream)
  Axi4StreamSpecRenamer(io.rs_stream)
  Axi4StreamSpecRenamer(io.v_stream)
  Axi4StreamSpecRenamer(io.vq_cache_i_stream)
  Axi4StreamSpecRenamer(io.vs_cache_i_stream)
  Axi4StreamSpecRenamer(io.vq_cache_o_stream)
  Axi4StreamSpecRenamer(io.vs_cache_o_stream)
  Axi4StreamSpecRenamer(io.aq_stream)
  Axi4StreamSpecRenamer(io.as_stream)
}

class SILU_GELU_Blackbox extends BlackBox {
  val top_name: String = "SILU_GELU"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)

    val ap_ctrl:      ApChain      = slave (ApChain())
    val mlp_i_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("mlp_i_stream", verilog_file_path)))
    val q_stream:     BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("q_stream",     verilog_file_path)))
    val s_stream:     BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("s_stream",     verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class SILU_GELU extends Component {
  setDefinitionName("SILU_GELU_wrapper")
  private val black_box = new SILU_GELU_Blackbox
  val io = new Bundle {
    val signals:      DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val mlp_i_stream: Axi4Stream = slave (Axi4Stream(black_box.io.mlp_i_stream.config.to_std_config()))
    val q_stream:     Axi4Stream = master(Axi4Stream(black_box.io.q_stream.config.to_std_config()))
    val s_stream:     Axi4Stream = master(Axi4Stream(black_box.io.s_stream.config.to_std_config()))
    val idle:         Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_SILU_GELU)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode    := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin := manager.io.signals.O.L_BEGIN
  black_box.io.l_close := manager.io.signals.O.L_CLOSE

  black_box.io.mlp_i_stream.connect2std(io.mlp_i_stream)
  black_box.io.q_stream    .connect2std(io.q_stream)
  black_box.io.s_stream    .connect2std(io.s_stream)
  Axi4StreamSpecRenamer(io.mlp_i_stream)
  Axi4StreamSpecRenamer(io.q_stream)
  Axi4StreamSpecRenamer(io.s_stream)
}

class RMS_LAYERNORM_Blackbox extends BlackBox {
  val top_name: String = "RMS_LAYERNORM"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)

    val ap_ctrl:    ApChain      = slave (ApChain())
    val x_stream:   BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("x_stream",    verilog_file_path)))
    val lnw_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("lnw_stream",  verilog_file_path)))
    val lnb_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("lnb_stream",  verilog_file_path)))
    val xlnq_stream:BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("xlnq_stream", verilog_file_path)))
    val xlns_stream:BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("xlns_stream", verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class RMS_LAYERNORM extends Component {
  setDefinitionName("RMS_LAYERNORM_wrapper")
  private val black_box = new RMS_LAYERNORM_Blackbox
  val io = new Bundle {
    val signals:     DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val x_stream:    Axi4Stream = slave (Axi4Stream(black_box.io.x_stream.config.to_std_config()))
    val lnw_stream:  Axi4Stream = slave (Axi4Stream(black_box.io.lnw_stream.config.to_std_config()))
    val lnb_stream:  Axi4Stream = slave (Axi4Stream(black_box.io.lnb_stream.config.to_std_config()))
    val xlnq_stream: Axi4Stream = master(Axi4Stream(black_box.io.xlnq_stream.config.to_std_config()))
    val xlns_stream: Axi4Stream = master(Axi4Stream(black_box.io.xlns_stream.config.to_std_config()))
    val idle:        Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_RMS_LAYERNORM)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode    := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin := manager.io.signals.O.L_BEGIN
  black_box.io.l_close := manager.io.signals.O.L_CLOSE

  black_box.io.x_stream   .connect2std(io.x_stream)
  black_box.io.lnw_stream .connect2std(io.lnw_stream)
  black_box.io.lnb_stream .connect2std(io.lnb_stream)
  black_box.io.xlnq_stream.connect2std(io.xlnq_stream)
  black_box.io.xlns_stream.connect2std(io.xlns_stream)
  Axi4StreamSpecRenamer(io.x_stream)
  Axi4StreamSpecRenamer(io.lnw_stream)
  Axi4StreamSpecRenamer(io.lnb_stream)
  Axi4StreamSpecRenamer(io.xlnq_stream)
  Axi4StreamSpecRenamer(io.xlns_stream)
}

class RESIDUAL_Blackbox extends BlackBox {
  val top_name: String = "RESIDUAL"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)

    val ap_ctrl:    ApChain      = slave (ApChain())
    val x_stream:   BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("x_stream",     verilog_file_path)))
    val res_i_stream:BlackboxAxis= slave (BlackboxAxis(BlackboxAxisConfig("res_i_stream", verilog_file_path)))
    val res_o_stream:BlackboxAxis= master(BlackboxAxis(BlackboxAxisConfig("res_o_stream", verilog_file_path)))
    val y_stream:   BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("y_stream",     verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class RESIDUAL extends Component {
  setDefinitionName("RESIDUAL_wrapper")
  private val black_box = new RESIDUAL_Blackbox
  val io = new Bundle {
    val signals:      DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val x_stream:     Axi4Stream = slave (Axi4Stream(black_box.io.x_stream.config.to_std_config()))
    val res_i_stream: Axi4Stream = slave (Axi4Stream(black_box.io.res_i_stream.config.to_std_config()))
    val res_o_stream: Axi4Stream = master(Axi4Stream(black_box.io.res_o_stream.config.to_std_config()))
    val y_stream:     Axi4Stream = master(Axi4Stream(black_box.io.y_stream.config.to_std_config()))
    val idle:         Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_RESIDUAL)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode    := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin := manager.io.signals.O.L_BEGIN
  black_box.io.l_close := manager.io.signals.O.L_CLOSE

  black_box.io.x_stream    .connect2std(io.x_stream)
  black_box.io.res_i_stream.connect2std(io.res_i_stream)
  black_box.io.res_o_stream.connect2std(io.res_o_stream)
  black_box.io.y_stream    .connect2std(io.y_stream)
  Axi4StreamSpecRenamer(io.x_stream)
  Axi4StreamSpecRenamer(io.res_i_stream)
  Axi4StreamSpecRenamer(io.res_o_stream)
  Axi4StreamSpecRenamer(io.y_stream)
}

class MUX_Blackbox extends BlackBox {
  val top_name: String = "MUX"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)

    val ap_ctrl:    ApChain      = slave (ApChain())
    val xlnq_stream:BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("xlnq_stream", verilog_file_path)))
    val xlns_stream:BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("xlns_stream", verilog_file_path)))
    val aq_stream:  BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("aq_stream",   verilog_file_path)))
    val as_stream:  BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("as_stream",   verilog_file_path)))
    val xmq_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("xmq_stream",  verilog_file_path)))
    val xms_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("xms_stream",  verilog_file_path)))
    val q_stream:   BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("q_stream",    verilog_file_path)))
    val s_stream:   BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("s_stream",    verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class MUX extends Component {
  setDefinitionName("MUX_wrapper")
  private val black_box = new MUX_Blackbox
  val io = new Bundle {
    val signals:     DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val xlnq_stream: Axi4Stream = slave (Axi4Stream(black_box.io.xlnq_stream.config.to_std_config()))
    val xlns_stream: Axi4Stream = slave (Axi4Stream(black_box.io.xlns_stream.config.to_std_config()))
    val aq_stream:   Axi4Stream = slave (Axi4Stream(black_box.io.aq_stream.config.to_std_config()))
    val as_stream:   Axi4Stream = slave (Axi4Stream(black_box.io.as_stream.config.to_std_config()))
    val xmq_stream:  Axi4Stream = slave (Axi4Stream(black_box.io.xmq_stream.config.to_std_config()))
    val xms_stream:  Axi4Stream = slave (Axi4Stream(black_box.io.xms_stream.config.to_std_config()))
    val q_stream:    Axi4Stream = master(Axi4Stream(black_box.io.q_stream.config.to_std_config()))
    val s_stream:    Axi4Stream = master(Axi4Stream(black_box.io.s_stream.config.to_std_config()))
    val idle:        Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_MUX)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode    := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin := manager.io.signals.O.L_BEGIN
  black_box.io.l_close := manager.io.signals.O.L_CLOSE

  black_box.io.xlnq_stream.connect2std(io.xlnq_stream)
  black_box.io.xlns_stream.connect2std(io.xlns_stream)
  black_box.io.aq_stream  .connect2std(io.aq_stream)
  black_box.io.as_stream  .connect2std(io.as_stream)
  black_box.io.xmq_stream .connect2std(io.xmq_stream)
  black_box.io.xms_stream .connect2std(io.xms_stream)
  black_box.io.q_stream   .connect2std(io.q_stream)
  black_box.io.s_stream   .connect2std(io.s_stream)
  Axi4StreamSpecRenamer(io.xlnq_stream)
  Axi4StreamSpecRenamer(io.xlns_stream)
  Axi4StreamSpecRenamer(io.aq_stream)
  Axi4StreamSpecRenamer(io.as_stream)
  Axi4StreamSpecRenamer(io.xmq_stream)
  Axi4StreamSpecRenamer(io.xms_stream)
  Axi4StreamSpecRenamer(io.q_stream)
  Axi4StreamSpecRenamer(io.s_stream)
}

class A_REORDER_Blackbox extends BlackBox {
  val top_name: String = "A_REORDER"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val memory_vit_a: UInt = in UInt(64 bits)

    val ap_ctrl:      ApChain      = slave (ApChain())
    val m_axi_gmem1:  BlackboxAxi  = master(BlackboxAxi(BlackboxAxiConfig("gmem1", verilog_file_path)))
    val rv_aq_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("rv_aq_stream",  verilog_file_path)))
    val rv_as_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("rv_as_stream",  verilog_file_path)))
    val mux_aq_stream:BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("mux_aq_stream", verilog_file_path)))
    val mux_as_stream:BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("mux_as_stream", verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  BlackboxAxiRenamer(io.m_axi_gmem1)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class A_REORDER extends Component {
  setDefinitionName("A_REORDER_wrapper")
  private val black_box = new A_REORDER_Blackbox
  val io = new Bundle {
    val signals:       DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val gmem1:         Axi4       = master(Axi4(black_box.io.m_axi_gmem1.config.to_std_config()))
    val rv_aq_stream:  Axi4Stream = slave (Axi4Stream(black_box.io.rv_aq_stream .config.to_std_config()))
    val rv_as_stream:  Axi4Stream = slave (Axi4Stream(black_box.io.rv_as_stream .config.to_std_config()))
    val mux_aq_stream: Axi4Stream = master(Axi4Stream(black_box.io.mux_aq_stream.config.to_std_config()))
    val mux_as_stream: Axi4Stream = master(Axi4Stream(black_box.io.mux_as_stream.config.to_std_config()))
    val idle:          Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_A_REORDER)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.memory_vit_a := manager.io.signals.O.MEMORY_VIT_A

  black_box.io.m_axi_gmem1 .connect2std(io.gmem1)
  black_box.io.rv_aq_stream .connect2std(io.rv_aq_stream)
  black_box.io.rv_as_stream .connect2std(io.rv_as_stream)
  black_box.io.mux_aq_stream.connect2std(io.mux_aq_stream)
  black_box.io.mux_as_stream.connect2std(io.mux_as_stream)
  Axi4SpecRenamer(io.gmem1)
  Axi4StreamSpecRenamer(io.rv_aq_stream)
  Axi4StreamSpecRenamer(io.rv_as_stream)
  Axi4StreamSpecRenamer(io.mux_aq_stream)
  Axi4StreamSpecRenamer(io.mux_as_stream)
}

class XM_REORDER_Blackbox extends BlackBox {
  val top_name: String = "XM_REORDER"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val memory_vit_xm: UInt = in UInt(64 bits)

    val ap_ctrl:       ApChain      = slave (ApChain())
    val m_axi_gmem1:   BlackboxAxi  = master(BlackboxAxi(BlackboxAxiConfig("gmem1", verilog_file_path)))
    val silu_q_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("silu_q_stream", verilog_file_path)))
    val silu_s_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("silu_s_stream", verilog_file_path)))
    val mux_q_stream:  BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("mux_q_stream",  verilog_file_path)))
    val mux_s_stream:  BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("mux_s_stream",  verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  BlackboxAxiRenamer(io.m_axi_gmem1)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class XM_REORDER extends Component {
  setDefinitionName("XM_REORDER_wrapper")
  private val black_box = new XM_REORDER_Blackbox
  val io = new Bundle {
    val signals:       DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val gmem1:         Axi4       = master(Axi4(black_box.io.m_axi_gmem1.config.to_std_config()))
    val silu_q_stream: Axi4Stream = slave (Axi4Stream(black_box.io.silu_q_stream.config.to_std_config()))
    val silu_s_stream: Axi4Stream = slave (Axi4Stream(black_box.io.silu_s_stream.config.to_std_config()))
    val mux_q_stream:  Axi4Stream = master(Axi4Stream(black_box.io.mux_q_stream .config.to_std_config()))
    val mux_s_stream:  Axi4Stream = master(Axi4Stream(black_box.io.mux_s_stream .config.to_std_config()))
    val idle:          Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_XM_REORDER)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.memory_vit_xm := manager.io.signals.O.MEMORY_VIT_XM

  black_box.io.m_axi_gmem1  .connect2std(io.gmem1)
  black_box.io.silu_q_stream.connect2std(io.silu_q_stream)
  black_box.io.silu_s_stream.connect2std(io.silu_s_stream)
  black_box.io.mux_q_stream .connect2std(io.mux_q_stream)
  black_box.io.mux_s_stream .connect2std(io.mux_s_stream)
  Axi4SpecRenamer(io.gmem1)
  Axi4StreamSpecRenamer(io.silu_q_stream)
  Axi4StreamSpecRenamer(io.silu_s_stream)
  Axi4StreamSpecRenamer(io.mux_q_stream)
  Axi4StreamSpecRenamer(io.mux_s_stream)
}

class M_AXI_Blackbox extends BlackBox {
  val top_name: String = "M_AXI"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val op:       UInt = in UInt(1 bits)

    val memory_vit_bias: UInt = in UInt(64 bits)
    val memory_llm_lnw:  UInt = in UInt(64 bits)
    val memory_cls_lnw:  UInt = in UInt(64 bits)
    val memory_vit_lnw:  UInt = in UInt(64 bits)
    val memory_vit_lnb:  UInt = in UInt(64 bits)

    val ap_ctrl:    ApChain      = slave (ApChain())
    val m_axi_gmem1:BlackboxAxi  = master(BlackboxAxi(BlackboxAxiConfig("gmem1", verilog_file_path)))
    val bias_stream:BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("bias_stream", verilog_file_path)))
    val lnw_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("lnw_stream",  verilog_file_path)))
    val lnb_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("lnb_stream",  verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  BlackboxAxiRenamer(io.m_axi_gmem1)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class M_AXI extends Component {
  setDefinitionName("M_AXI_wrapper")
  private val black_box = new M_AXI_Blackbox
  val io = new Bundle {
    val signals:    DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val gmem1:      Axi4       = master(Axi4(black_box.io.m_axi_gmem1.config.to_std_config()))
    val bias_stream:Axi4Stream = master(Axi4Stream(black_box.io.bias_stream.config.to_std_config()))
    val lnw_stream: Axi4Stream = master(Axi4Stream(black_box.io.lnw_stream.config.to_std_config()))
    val lnb_stream: Axi4Stream = master(Axi4Stream(black_box.io.lnb_stream.config.to_std_config()))
    val idle:       Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_M_AXI)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode            := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin         := manager.io.signals.O.L_BEGIN
  black_box.io.l_close         := manager.io.signals.O.L_CLOSE
  black_box.io.op              := manager.io.signals.O.PARAM_OP.resized
  black_box.io.memory_vit_bias := manager.io.signals.O.MEMORY_VIT_BIAS
  black_box.io.memory_llm_lnw  := manager.io.signals.O.MEMORY_LLM_LNW
  black_box.io.memory_cls_lnw  := manager.io.signals.O.MEMORY_CLS_LNW
  black_box.io.memory_vit_lnw  := manager.io.signals.O.MEMORY_VIT_LNW
  black_box.io.memory_vit_lnb  := manager.io.signals.O.MEMORY_VIT_LNB

  black_box.io.m_axi_gmem1.connect2std(io.gmem1)
  black_box.io.bias_stream.connect2std(io.bias_stream)
  black_box.io.lnw_stream .connect2std(io.lnw_stream)
  black_box.io.lnb_stream .connect2std(io.lnb_stream)
  Axi4SpecRenamer(io.gmem1)
  Axi4StreamSpecRenamer(io.bias_stream)
  Axi4StreamSpecRenamer(io.lnw_stream)
  Axi4StreamSpecRenamer(io.lnb_stream)
}

class WEIGHT_AXI_Blackbox extends BlackBox {
  val top_name: String = "WEIGHT_AXI"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)

    val memory_decoder_w_lo: UInt = in UInt(64 bits)
    val memory_decoder_w_hi: UInt = in UInt(64 bits)
    val memory_cls_w_lo:     UInt = in UInt(64 bits)
    val memory_cls_w_hi:     UInt = in UInt(64 bits)
    val memory_vit_w_lo:     UInt = in UInt(64 bits)
    val memory_vit_w_hi:     UInt = in UInt(64 bits)

    val ap_ctrl:     ApChain      = slave (ApChain())
    val m_axi_gmem1: BlackboxAxi  = master(BlackboxAxi(BlackboxAxiConfig("gmem1", verilog_file_path)))
    val m_axi_gmem2: BlackboxAxi  = master(BlackboxAxi(BlackboxAxiConfig("gmem2", verilog_file_path)))
    val wq_stream:   BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("wq_stream",  verilog_file_path)))
    val ws1_stream:  BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("ws1_stream", verilog_file_path)))
    val ws2_stream:  BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("ws2_stream", verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  BlackboxAxiRenamer(io.m_axi_gmem1)
  BlackboxAxiRenamer(io.m_axi_gmem2)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class WEIGHT_AXI extends Component {
  setDefinitionName("WEIGHT_AXI_wrapper")
  private val black_box = new WEIGHT_AXI_Blackbox
  val io = new Bundle {
    val signals:   DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val gmem1:     Axi4       = master(Axi4(black_box.io.m_axi_gmem1.config.to_std_config()))
    val gmem2:     Axi4       = master(Axi4(black_box.io.m_axi_gmem2.config.to_std_config()))
    val wq_stream: Axi4Stream = master(Axi4Stream(black_box.io.wq_stream.config.to_std_config()))
    val ws1_stream:Axi4Stream = master(Axi4Stream(black_box.io.ws1_stream.config.to_std_config()))
    val ws2_stream:Axi4Stream = master(Axi4Stream(black_box.io.ws2_stream.config.to_std_config()))
    val idle:      Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_WEIGHT_AXI)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode                := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin             := manager.io.signals.O.L_BEGIN
  black_box.io.l_close             := manager.io.signals.O.L_CLOSE
  black_box.io.memory_decoder_w_lo := manager.io.signals.O.MEMORY_DECODER_W_LO
  black_box.io.memory_decoder_w_hi := manager.io.signals.O.MEMORY_DECODER_W_HI
  black_box.io.memory_cls_w_lo     := manager.io.signals.O.MEMORY_CLS_W_LO
  black_box.io.memory_cls_w_hi     := manager.io.signals.O.MEMORY_CLS_W_HI
  black_box.io.memory_vit_w_lo     := manager.io.signals.O.MEMORY_VIT_W_LO
  black_box.io.memory_vit_w_hi     := manager.io.signals.O.MEMORY_VIT_W_HI

  black_box.io.m_axi_gmem1.connect2std(io.gmem1)
  black_box.io.m_axi_gmem2.connect2std(io.gmem2)
  black_box.io.wq_stream  .connect2std(io.wq_stream)
  black_box.io.ws1_stream .connect2std(io.ws1_stream)
  black_box.io.ws2_stream .connect2std(io.ws2_stream)
  Axi4SpecRenamer(io.gmem1)
  Axi4SpecRenamer(io.gmem2)
  Axi4StreamSpecRenamer(io.wq_stream)
  Axi4StreamSpecRenamer(io.ws1_stream)
  Axi4StreamSpecRenamer(io.ws2_stream)
}

class STATE_AXI_Blackbox extends BlackBox {
  val top_name: String = "STATE_AXI"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val op:       UInt = in UInt(3 bits)

    val memory_decoder_state: UInt = in UInt(64 bits)
    val memory_vit_state:     UInt = in UInt(64 bits)
    val memory_cls_y:         UInt = in UInt(64 bits)

    val ap_ctrl:     ApChain      = slave (ApChain())
    val m_axi_gmem1: BlackboxAxi  = master(BlackboxAxi(BlackboxAxiConfig("gmem1", verilog_file_path)))
    val x_stream:    BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("x_stream",   verilog_file_path)))
    val y_stream:    BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("y_stream",   verilog_file_path)))
    val cls_stream:  BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("cls_stream", verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  BlackboxAxiRenamer(io.m_axi_gmem1)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class STATE_AXI extends Component {
  setDefinitionName("STATE_AXI_wrapper")
  private val black_box = new STATE_AXI_Blackbox
  val io = new Bundle {
    val signals:    DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val gmem1:      Axi4       = master(Axi4(black_box.io.m_axi_gmem1.config.to_std_config()))
    val x_stream:   Axi4Stream = master(Axi4Stream(black_box.io.x_stream.config.to_std_config()))
    val y_stream:   Axi4Stream = slave (Axi4Stream(black_box.io.y_stream.config.to_std_config()))
    val cls_stream: Axi4Stream = slave (Axi4Stream(black_box.io.cls_stream.config.to_std_config()))
    val idle:       Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_STATE_AXI)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode                 := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin              := manager.io.signals.O.L_BEGIN
  black_box.io.l_close              := manager.io.signals.O.L_CLOSE
  black_box.io.op                   := manager.io.signals.O.STATE_OP.resized
  black_box.io.memory_decoder_state := manager.io.signals.O.MEMORY_DECODER_STATE
  black_box.io.memory_vit_state     := manager.io.signals.O.MEMORY_VIT_STATE
  black_box.io.memory_cls_y         := manager.io.signals.O.MEMORY_CLS_Y

  black_box.io.m_axi_gmem1.connect2std(io.gmem1)
  black_box.io.x_stream   .connect2std(io.x_stream)
  black_box.io.y_stream   .connect2std(io.y_stream)
  black_box.io.cls_stream .connect2std(io.cls_stream)
  Axi4SpecRenamer(io.gmem1)
  Axi4StreamSpecRenamer(io.x_stream)
  Axi4StreamSpecRenamer(io.y_stream)
  Axi4StreamSpecRenamer(io.cls_stream)
}

class KV_CACHE_Blackbox extends BlackBox {
  val top_name: String = "KV_CACHE"
  setDefinitionName(top_name)
  private val verilog_file_path: String = s"src/main/verilog/$top_name/all.v"
  val io = new Bundle {
    val ap_clk:   Bool = in Bool()
    val ap_rst_n: Bool = in Bool()
    val mode:     Bool = in Bool()
    val l_begin:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val l_close:  UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val pos_r:    UInt = in UInt(ctrl_cfg.REG_BITS bits)
    val memory_k_cache: UInt = in UInt(64 bits)

    val ap_ctrl:           ApChain      = slave (ApChain())
    val m_axi_gmem1:       BlackboxAxi  = master(BlackboxAxi(BlackboxAxiConfig("gmem1", verilog_file_path)))
    val kq_cache_i_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("kq_cache_i_stream", verilog_file_path)))
    val kq_cache_o_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("kq_cache_o_stream", verilog_file_path)))
    val ks_cache_i_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("ks_cache_i_stream", verilog_file_path)))
    val ks_cache_o_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("ks_cache_o_stream", verilog_file_path)))
    val vq_cache_i_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("vq_cache_i_stream", verilog_file_path)))
    val vq_cache_o_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("vq_cache_o_stream", verilog_file_path)))
    val vs_cache_i_stream: BlackboxAxis = master(BlackboxAxis(BlackboxAxisConfig("vs_cache_i_stream", verilog_file_path)))
    val vs_cache_o_stream: BlackboxAxis = slave (BlackboxAxis(BlackboxAxisConfig("vs_cache_o_stream", verilog_file_path)))
  }
  noIoPrefix()
  ApChainRenamer(io.ap_ctrl)
  BlackboxAxiRenamer(io.m_axi_gmem1)
  mapClockDomain(clock = io.ap_clk, reset = io.ap_rst_n, resetActiveLevel = LOW)
  addRTLPath(verilog_file_path)
}

class KV_CACHE extends Component {
  setDefinitionName("KV_CACHE_wrapper")
  private val black_box = new KV_CACHE_Blackbox
  val io = new Bundle {
    val signals:           DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val gmem1:             Axi4       = master(Axi4(black_box.io.m_axi_gmem1.config.to_std_config()))
    val kq_cache_i_stream: Axi4Stream = master(Axi4Stream(black_box.io.kq_cache_i_stream.config.to_std_config()))
    val kq_cache_o_stream: Axi4Stream = slave (Axi4Stream(black_box.io.kq_cache_o_stream.config.to_std_config()))
    val ks_cache_i_stream: Axi4Stream = master(Axi4Stream(black_box.io.ks_cache_i_stream.config.to_std_config()))
    val ks_cache_o_stream: Axi4Stream = slave (Axi4Stream(black_box.io.ks_cache_o_stream.config.to_std_config()))
    val vq_cache_i_stream: Axi4Stream = master(Axi4Stream(black_box.io.vq_cache_i_stream.config.to_std_config()))
    val vq_cache_o_stream: Axi4Stream = slave (Axi4Stream(black_box.io.vq_cache_o_stream.config.to_std_config()))
    val vs_cache_i_stream: Axi4Stream = master(Axi4Stream(black_box.io.vs_cache_i_stream.config.to_std_config()))
    val vs_cache_o_stream: Axi4Stream = slave (Axi4Stream(black_box.io.vs_cache_o_stream.config.to_std_config()))
    val idle:              Bool       = out Bool()
  }
  noIoPrefix()
  io.idle := black_box.io.ap_ctrl.ap_idle
  val manager = new Manager(ctrl_cfg.IP_KV_CACHE)
  manager.io.signals <> io.signals
  manager.io.ap_ctrl <> black_box.io.ap_ctrl
  black_box.io.mode           := manager.io.signals.O.MODE.asBool
  black_box.io.l_begin        := manager.io.signals.O.L_BEGIN
  black_box.io.l_close        := manager.io.signals.O.L_CLOSE
  black_box.io.pos_r          := manager.io.signals.O.POS
  black_box.io.memory_k_cache := manager.io.signals.O.MEMORY_K_CACHE

  black_box.io.m_axi_gmem1      .connect2std(io.gmem1)
  black_box.io.kq_cache_i_stream.connect2std(io.kq_cache_i_stream)
  black_box.io.kq_cache_o_stream.connect2std(io.kq_cache_o_stream)
  black_box.io.ks_cache_i_stream.connect2std(io.ks_cache_i_stream)
  black_box.io.ks_cache_o_stream.connect2std(io.ks_cache_o_stream)
  black_box.io.vq_cache_i_stream.connect2std(io.vq_cache_i_stream)
  black_box.io.vq_cache_o_stream.connect2std(io.vq_cache_o_stream)
  black_box.io.vs_cache_i_stream.connect2std(io.vs_cache_i_stream)
  black_box.io.vs_cache_o_stream.connect2std(io.vs_cache_o_stream)
  Axi4SpecRenamer(io.gmem1)
  Axi4StreamSpecRenamer(io.kq_cache_i_stream)
  Axi4StreamSpecRenamer(io.kq_cache_o_stream)
  Axi4StreamSpecRenamer(io.ks_cache_i_stream)
  Axi4StreamSpecRenamer(io.ks_cache_o_stream)
  Axi4StreamSpecRenamer(io.vq_cache_i_stream)
  Axi4StreamSpecRenamer(io.vq_cache_o_stream)
  Axi4StreamSpecRenamer(io.vs_cache_i_stream)
  Axi4StreamSpecRenamer(io.vs_cache_o_stream)
}

// @formatter:on
