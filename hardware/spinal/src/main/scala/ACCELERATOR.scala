import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.bus.amba4.axi._
import spinal.lib.bus.amba4.axis._
import spinal.lib.bus.amba4.axilite.{AxiLite4, AxiLite4SpecRenamer}
import spinal.lib.bus.amba4.axis.Axi4Stream.Axi4Stream
import utils._

import scala.language.postfixOps

// @formatter:off

class ACCELERATOR extends Component {
  // 复用版顶层只实例化一套物理 IP。mode/op/layer/pos/地址由 Controller 寄存器稳定广播，
  // RUN_MASK 允许 PYNQ 按阶段启动参数、权重、state、KV cache 或计算链。
  // 这些实例不增加顶层 IO；保持为公开 val 只是为了 Spinal 仿真 harness 能读取
  // 各 IP idle 和内部 AXIS 握手计数，定位 decoder layer 长 payload 的反压位置。
  val inst_controller    = new Controller
  val inst_m_axi         = new M_AXI
  val inst_weight_axi    = new WEIGHT_AXI
  val inst_kv_cache      = new KV_CACHE
  val inst_state_axi     = new STATE_AXI

  val inst_mux           = new MUX
  val inst_permute       = new PERMUTE
  val inst_demux         = new DEMUX
  val inst_rope_qk       = new ROPE_QK
  val inst_qk_gemm       = new QK_GEMM
  val inst_softmax       = new SOFTMAX
  val inst_rv_gemm       = new RV_GEMM
  val inst_a_reorder     = new A_REORDER
  val inst_silu_gelu     = new SILU_GELU
  val inst_xm_reorder    = new XM_REORDER
  val inst_residual      = new RESIDUAL
  val inst_rms_layernorm = new RMS_LAYERNORM

  // mode 来自 Controller 寄存器，一次 RUN_MASK 调用期间保持稳定。这里仅用于
  // 顶层 AXIS 旁路选择：LLM 不需要 A/XM scratch，ViT 才经过 reorder IP。
  private val vitMode = inst_controller.io.signals.MODE.asBool

  private def splitAxisByMode(src: Axi4Stream, vitDst: Axi4Stream, llmDst: Axi4Stream): Unit = {
    src.ready    := Mux(vitMode, vitDst.ready, llmDst.ready)
    vitDst.valid := src.valid && vitMode
    llmDst.valid := src.valid && !vitMode
    vitDst.data  := src.data
    llmDst.data  := src.data
  }

  private def selectAxisByMode(vitSrc: Axi4Stream, llmSrc: Axi4Stream, dst: Axi4Stream): Unit = {
    dst.valid    := Mux(vitMode, vitSrc.valid, llmSrc.valid)
    dst.data     := Mux(vitMode, vitSrc.data, llmSrc.data)
    vitSrc.ready := dst.ready && vitMode
    llmSrc.ready := dst.ready && !vitMode
  }

  // 控制链只传播稳定配置和启动脉冲；每个 wrapper 用自己的 RUN_MASK bit 决定是否启动。
  inst_controller    .io.signals      <> inst_m_axi        .io.signals.I
  inst_m_axi         .io.signals.O    <> inst_weight_axi   .io.signals.I
  inst_weight_axi    .io.signals.O    <> inst_kv_cache     .io.signals.I
  inst_kv_cache      .io.signals.O    <> inst_state_axi    .io.signals.I
  inst_state_axi     .io.signals.O    <> inst_mux          .io.signals.I
  inst_mux           .io.signals.O    <> inst_permute      .io.signals.I
  inst_permute       .io.signals.O    <> inst_demux        .io.signals.I
  inst_demux         .io.signals.O    <> inst_rope_qk      .io.signals.I
  inst_rope_qk       .io.signals.O    <> inst_qk_gemm      .io.signals.I
  inst_qk_gemm       .io.signals.O    <> inst_softmax      .io.signals.I
  inst_softmax       .io.signals.O    <> inst_rv_gemm      .io.signals.I
  inst_rv_gemm       .io.signals.O    <> inst_a_reorder    .io.signals.I
  inst_a_reorder     .io.signals.O    <> inst_silu_gelu    .io.signals.I
  inst_silu_gelu     .io.signals.O    <> inst_xm_reorder   .io.signals.I
  inst_xm_reorder    .io.signals.O    <> inst_residual     .io.signals.I
  inst_residual      .io.signals.O    <> inst_rms_layernorm.io.signals.I

  // 参数流：M_AXI 只喂 ViT bias 和 RMS/LayerNorm gamma/beta，LLM mode 不消费 bias/lnb。
  inst_m_axi.io.bias_stream.queue(64) <> inst_demux.io.bias_stream
  // ViT 顶层调度先单独启动 M_AXI 预取一层两次 LayerNorm 的 gamma/beta，
  // 随后主链启动时同一套 M_AXI 改为发送 DEMUX bias；256 拍可容纳 ViT 2*768/8=192 拍参数。
  inst_m_axi.io.lnw_stream .queue(256) <> inst_rms_layernorm.io.lnw_stream
  inst_m_axi.io.lnb_stream .queue(256) <> inst_rms_layernorm.io.lnb_stream

  // 权重流：WEIGHT_AXI 的 compact WQ/WS 输出接入共享 PERMUTE tensor core。
  inst_weight_axi.io.wq_stream .queue(256) <> inst_permute.io.w_stream
  inst_weight_axi.io.ws1_stream.queue(64)  <> inst_permute.io.s1_stream
  inst_weight_axi.io.ws2_stream.queue(64)  <> inst_permute.io.s2_stream

  // 主计算链：MUX -> PERMUTE -> DEMUX -> attention/MLP/residual/norm，再反馈到 MUX。
  inst_mux    .io.q_stream <> inst_permute.io.i_stream
  inst_mux    .io.s_stream.queue(16) <> inst_permute.io.s_stream
  inst_permute.io.o_stream <> inst_demux.io.gemm_stream

  inst_demux  .io.qk_stream     <> inst_rope_qk  .io.qk_i_stream
  inst_rope_qk.io.qk_q_stream   <> inst_qk_gemm  .io.qk_q_stream
  inst_rope_qk.io.qk_s_stream   <> inst_qk_gemm  .io.qk_s_stream
  inst_qk_gemm.io.r_stream      <> inst_softmax  .io.r_stream
  inst_softmax.io.rq_stream     <> inst_rv_gemm  .io.rq_stream
  inst_softmax.io.rs_stream     <> inst_rv_gemm  .io.rs_stream
  inst_demux  .io.v_stream      <> inst_rv_gemm  .io.v_stream

  inst_demux  .io.mlp1_stream   <> inst_silu_gelu.io.mlp_i_stream
  // RESIDUAL 的 state mover 采用“发 delta-order x tile 后等待 y writeback”的闭合调度。
  // O/FC2 delta 需要先完整穿过 GEMM/DEMUX，避免 RESIDUAL 暂时不 ready 时反压上游反馈链。
  inst_demux  .io.od_fc2_stream.queue(1024) <> inst_residual .io.res_i_stream
  inst_demux  .io.cls_stream    <> inst_state_axi.io.cls_stream

  inst_state_axi.io.x_stream    <> inst_residual.io.x_stream
  inst_residual .io.res_o_stream<> inst_rms_layernorm.io.x_stream
  inst_residual .io.y_stream    <> inst_state_axi.io.y_stream

  // ViT RV_GEMM 的 A 自然顺序是 H -> token -> HCT，而 MUX 的 O 投影按
  // TT -> H -> TP -> HCT 消费。A_REORDER 用外部 DDR 做 scratch；LLM 的 8-token
  // A 顺序已匹配 MUX，直接旁路，不启动 A_REORDER，避免无效 IP 资源/等待。
  val llm_aq_bypass = Axi4Stream(inst_rv_gemm.io.aq_stream.config)
  val llm_as_bypass = Axi4Stream(inst_rv_gemm.io.as_stream.config)
  splitAxisByMode(inst_rv_gemm.io.aq_stream, inst_a_reorder.io.rv_aq_stream, llm_aq_bypass)
  splitAxisByMode(inst_rv_gemm.io.as_stream, inst_a_reorder.io.rv_as_stream, llm_as_bypass)
  selectAxisByMode(inst_a_reorder.io.mux_aq_stream, llm_aq_bypass, inst_mux.io.aq_stream)
  selectAxisByMode(inst_a_reorder.io.mux_as_stream, llm_as_bypass, inst_mux.io.as_stream)
  // ViT MLP 中 SILU_GELU/GELU 会在 MUX 仍处于 FC1 replay 时提前产生 XM。
  // XM_REORDER 使用外部 scratch 做阶段解耦；LLM XM 直接旁路给 MUX。
  val llm_xmq_bypass = Axi4Stream(inst_silu_gelu.io.q_stream.config)
  val llm_xms_bypass = Axi4Stream(inst_silu_gelu.io.s_stream.config)
  splitAxisByMode(inst_silu_gelu.io.q_stream, inst_xm_reorder.io.silu_q_stream, llm_xmq_bypass)
  splitAxisByMode(inst_silu_gelu.io.s_stream, inst_xm_reorder.io.silu_s_stream, llm_xms_bypass)
  selectAxisByMode(inst_xm_reorder.io.mux_q_stream, llm_xmq_bypass, inst_mux.io.xmq_stream)
  selectAxisByMode(inst_xm_reorder.io.mux_s_stream, llm_xms_bypass, inst_mux.io.xms_stream)
  inst_rms_layernorm.io.xlnq_stream <> inst_mux.io.xlnq_stream
  inst_rms_layernorm.io.xlns_stream <> inst_mux.io.xlns_stream

  // KV cache 只在 LLM mode 由 RUN_MASK 显式启动；ViT mode 下 HLS IP 自身 no-op。
  inst_kv_cache.io.kq_cache_i_stream <> inst_qk_gemm.io.kq_cache_i_stream
  inst_kv_cache.io.ks_cache_i_stream <> inst_qk_gemm.io.ks_cache_i_stream
  // QK/RV 会在计算中较早写出当前 8-token K/V tile；KV_CACHE 同时还在重放整层 cache。
  // 这里用 1024 拍 FIFO 容纳完整 current tile，避免 256 深度填满后反压 attention 主链。
  inst_qk_gemm.io.kq_cache_o_stream.queue(1024) <> inst_kv_cache.io.kq_cache_o_stream
  inst_qk_gemm.io.ks_cache_o_stream.queue(1024) <> inst_kv_cache.io.ks_cache_o_stream
  inst_kv_cache.io.vq_cache_i_stream <> inst_rv_gemm.io.vq_cache_i_stream
  inst_kv_cache.io.vs_cache_i_stream <> inst_rv_gemm.io.vs_cache_i_stream
  inst_rv_gemm.io.vq_cache_o_stream.queue(1024) <> inst_kv_cache.io.vq_cache_o_stream
  inst_rv_gemm.io.vs_cache_o_stream.queue(1024) <> inst_kv_cache.io.vs_cache_o_stream

  private def simPublicAxis(stream: Axi4Stream): Unit = {
    // 仅服务 Verilator debug：允许测试 harness 读取内部 ready/valid，
    // 以及按需读取 payload data 做数值 tap；不会改变 ACCELERATOR 的硬件 IO。
    stream.valid.simPublic()
    stream.ready.simPublic()
    stream.data.simPublic()
  }

  Seq(
    inst_m_axi.io.lnw_stream,
    inst_weight_axi.io.wq_stream,
    inst_weight_axi.io.ws1_stream,
    inst_weight_axi.io.ws2_stream,
    inst_kv_cache.io.kq_cache_i_stream,
    inst_kv_cache.io.ks_cache_i_stream,
    inst_kv_cache.io.vq_cache_i_stream,
    inst_kv_cache.io.vs_cache_i_stream,
    inst_qk_gemm.io.kq_cache_o_stream,
    inst_qk_gemm.io.ks_cache_o_stream,
    inst_rv_gemm.io.vq_cache_o_stream,
    inst_rv_gemm.io.vs_cache_o_stream,
    inst_state_axi.io.x_stream,
    inst_residual.io.y_stream,
    inst_demux.io.cls_stream,
    inst_rms_layernorm.io.xlnq_stream,
    inst_rms_layernorm.io.xlns_stream,
    inst_mux.io.q_stream,
    inst_mux.io.s_stream,
    inst_permute.io.o_stream,
    inst_demux.io.qk_stream,
    inst_demux.io.v_stream,
    inst_demux.io.mlp1_stream,
    inst_demux.io.od_fc2_stream,
    inst_rope_qk.io.qk_q_stream,
    inst_rope_qk.io.qk_s_stream,
    inst_qk_gemm.io.r_stream,
    inst_softmax.io.rq_stream,
    inst_softmax.io.rs_stream,
    inst_rv_gemm.io.aq_stream,
    inst_rv_gemm.io.as_stream,
    inst_a_reorder.io.mux_aq_stream,
    inst_a_reorder.io.mux_as_stream,
    inst_silu_gelu.io.q_stream,
    inst_silu_gelu.io.s_stream,
    inst_xm_reorder.io.mux_q_stream,
    inst_xm_reorder.io.mux_s_stream,
    inst_residual.io.res_o_stream
  ).foreach(simPublicAxis)

  Seq(
    inst_m_axi.io.idle,
    inst_weight_axi.io.idle,
    inst_kv_cache.io.idle,
    inst_state_axi.io.idle,
    inst_mux.io.idle,
    inst_permute.io.idle,
    inst_demux.io.idle,
    inst_rope_qk.io.idle,
    inst_qk_gemm.io.idle,
    inst_softmax.io.idle,
    inst_rv_gemm.io.idle,
    inst_a_reorder.io.idle,
    inst_silu_gelu.io.idle,
    inst_xm_reorder.io.idle,
    inst_residual.io.idle,
    inst_rms_layernorm.io.idle
  ).foreach(_.simPublic())

  private val allIdle =
    inst_m_axi.io.idle && inst_weight_axi.io.idle && inst_kv_cache.io.idle && inst_state_axi.io.idle &&
    inst_mux.io.idle && inst_permute.io.idle && inst_demux.io.idle && inst_rope_qk.io.idle &&
    inst_qk_gemm.io.idle && inst_softmax.io.idle && inst_rv_gemm.io.idle && inst_a_reorder.io.idle && inst_silu_gelu.io.idle && inst_xm_reorder.io.idle &&
    inst_residual.io.idle && inst_rms_layernorm.io.idle

  private val vitLayerRunMask =
    (1 << ctrl_cfg.IP_M_AXI) |
    (1 << ctrl_cfg.IP_WEIGHT_AXI) |
    (1 << ctrl_cfg.IP_STATE_AXI) |
    (1 << ctrl_cfg.IP_MUX) |
    (1 << ctrl_cfg.IP_PERMUTE) |
    (1 << ctrl_cfg.IP_DEMUX) |
    (1 << ctrl_cfg.IP_ROPE_QK) |
    (1 << ctrl_cfg.IP_QK_GEMM) |
    (1 << ctrl_cfg.IP_SOFTMAX) |
    (1 << ctrl_cfg.IP_RV_GEMM) |
    (1 << ctrl_cfg.IP_A_REORDER) |
    (1 << ctrl_cfg.IP_SILU_GELU) |
    (1 << ctrl_cfg.IP_XM_REORDER) |
    (1 << ctrl_cfg.IP_RESIDUAL) |
    (1 << ctrl_cfg.IP_RMS_LAYERNORM)

  val inst_vit_stage_probe = new VitStageProbe
  inst_vit_stage_probe.io.launch := vitMode && inst_controller.io.signals.T &&
    (inst_controller.io.signals.RUN_MASK === U(vitLayerRunMask, ctrl_cfg.REG_BITS bits))
  inst_vit_stage_probe.io.allIdle := allIdle
  inst_vit_stage_probe.io.aWFire := inst_a_reorder.io.gmem1.w.valid && inst_a_reorder.io.gmem1.w.ready
  inst_vit_stage_probe.io.aBFire := inst_a_reorder.io.gmem1.b.valid && inst_a_reorder.io.gmem1.b.ready
  inst_vit_stage_probe.io.aArFire := inst_a_reorder.io.gmem1.ar.valid && inst_a_reorder.io.gmem1.ar.ready
  inst_vit_stage_probe.io.aRFire := inst_a_reorder.io.gmem1.r.valid && inst_a_reorder.io.gmem1.r.ready
  inst_vit_stage_probe.io.xmWFire := inst_xm_reorder.io.gmem1.w.valid && inst_xm_reorder.io.gmem1.w.ready
  inst_vit_stage_probe.io.xmBFire := inst_xm_reorder.io.gmem1.b.valid && inst_xm_reorder.io.gmem1.b.ready
  inst_vit_stage_probe.io.xmArFire := inst_xm_reorder.io.gmem1.ar.valid && inst_xm_reorder.io.gmem1.ar.ready
  inst_vit_stage_probe.io.xmRFire := inst_xm_reorder.io.gmem1.r.valid && inst_xm_reorder.io.gmem1.r.ready
  inst_controller.io.vitProbe := inst_vit_stage_probe.io.status

  val io = new Bundle {
    val axilite:      AxiLite4 = slave (AxiLite4(inst_controller.io.axilite.config))
    val param_gmem:   Axi4     = master(Axi4(inst_m_axi     .io.gmem1.config))
    val weight_gmem1: Axi4     = master(Axi4(inst_weight_axi.io.gmem1.config))
    val weight_gmem2: Axi4     = master(Axi4(inst_weight_axi.io.gmem2.config))
    val state_gmem:   Axi4     = master(Axi4(inst_state_axi .io.gmem1.config))
    val vit_a_gmem:   Axi4     = master(Axi4(inst_a_reorder .io.gmem1.config))
    val vit_xm_gmem:  Axi4     = master(Axi4(inst_xm_reorder.io.gmem1.config))
    val kv_gmem:      Axi4     = master(Axi4(inst_kv_cache  .io.gmem1.config))
    val idle:         Bool     = out Bool()
  }
  noIoPrefix()

  io.axilite      <> inst_controller.io.axilite
  io.param_gmem   <> inst_m_axi     .io.gmem1
  io.weight_gmem1 <> inst_weight_axi.io.gmem1
  io.weight_gmem2 <> inst_weight_axi.io.gmem2
  io.state_gmem   <> inst_state_axi .io.gmem1
  io.vit_a_gmem   <> inst_a_reorder .io.gmem1
  io.vit_xm_gmem  <> inst_xm_reorder.io.gmem1
  io.kv_gmem      <> inst_kv_cache  .io.gmem1
  io.idle         := allIdle
  inst_controller.io.idle := allIdle

  AxiLite4SpecRenamer(io.axilite)
  Axi4SpecRenamer(io.param_gmem)
  Axi4SpecRenamer(io.weight_gmem1)
  Axi4SpecRenamer(io.weight_gmem2)
  Axi4SpecRenamer(io.state_gmem)
  Axi4SpecRenamer(io.vit_a_gmem)
  Axi4SpecRenamer(io.vit_xm_gmem)
  Axi4SpecRenamer(io.kv_gmem)
}

object generate_accelerator extends App {
  // 生成 Verilog 时会合并当前复用 IP 黑盒路径；本轮按用户要求不执行生成/编译验证。
  val spinalConfig = SpinalConfig(
    defaultConfigForClockDomains = ClockDomainConfig(
      resetKind = ASYNC,
      resetActiveLevel = LOW
    ),
    mode = Verilog
  )
  SpinalVerilog(spinalConfig)(new ACCELERATOR).mergeRTLSource()
}

// @formatter:on
