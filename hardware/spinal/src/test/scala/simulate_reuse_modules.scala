import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import utils._

// @formatter:off

// 复用 IP 的单模块仿真入口。
// 当前默认做零层 smoke：l_begin == l_close，不喂真实 payload，只验证 wrapper、黑盒 RTL、
// Manager/RUN_MASK、reset/start/idle 握手能在 Verilator 下跑通。后续数据准备好后，
// 可以在对应 runXxx() 里把 Qwen 风格的 array2stream/stream2array 数值比对补上。
object ReuseModuleSmokeSim {
  private val timeoutCycles = sys.env.getOrElse("VLM_SINGLE_MODULE_TIMEOUT", "5000").toInt

  private val spinalConfig = SpinalConfig(
    defaultConfigForClockDomains = ClockDomainConfig(
      resetKind = SYNC,
      resetActiveLevel = LOW
    )
  )

  private def simConfig = {
    val base = SimConfig
      .withConfig(spinalConfig)
      .withWaveDepth(1)
      .allOptimisation
      .withVerilator
      .addSimulatorFlag("--unroll-count 1024")
      .addSimulatorFlag("-j 16")
      .addSimulatorFlag("-O3 --x-assign fast --x-initial fast --noassert")
    if (sys.env.get("VLM_SINGLE_MODULE_WAVE").contains("1")) base.withFstWave else base
  }

  private def initBaseSignals(signals: DaisyChain[ManagerSignals], ipBit: Int): Unit = {
    init_daisy_chain(signals)
    // 零层 smoke 不消费 AXIS/AXI payload；RUN_MASK 只启动当前 wrapper，避免误触发其它 IP。
    signals.I.MODE #= 1
    signals.I.L_BEGIN #= 0
    signals.I.L_CLOSE #= 0
    signals.I.POS #= 0
    signals.I.PARAM_OP #= 0
    signals.I.STATE_OP #= 0
    signals.I.RUN_MASK #= (BigInt(1) << ipBit)
  }

  private def launchZeroLayer(
    name: String,
    ipBit: Int,
    signals: DaisyChain[ManagerSignals],
    idle: Bool,
    clockDomain: ClockDomain
  ): Unit = {
    initBaseSignals(signals, ipBit)
    init_clock(clockDomain, 10)
    // reset 之后重新写一遍，避免测试激励被复位阶段的随机值覆盖。
    initBaseSignals(signals, ipBit)
    clockDomain.waitSampling(20)

    signals.I.T #= true
    clockDomain.waitSampling()
    signals.I.T #= false

    var cycles = 0
    while (!idle.toBoolean && cycles < timeoutCycles) {
      clockDomain.waitSampling()
      cycles += 1
    }
    assert(idle.toBoolean, s"$name did not return idle within $timeoutCycles cycles")
    clockDomain.waitSampling(10)
    println(s"$name single-wrapper smoke passed")
    simSuccess()
  }

  private def runComponent[T <: Component](name: String, gen: => T)(body: T => Unit): Unit = {
    println(s"Simulating single wrapper $name")
    simConfig.compile(gen).doSimUntilVoid(body)
  }

  private def runM_AXI(): Unit = runComponent("M_AXI", new M_AXI) { dut =>
    AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig()).reset()
    init_o_stream(dut.io.bias_stream)
    init_o_stream(dut.io.lnw_stream)
    init_o_stream(dut.io.lnb_stream)
    launchZeroLayer("M_AXI", ctrl_cfg.IP_M_AXI, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runWEIGHT_AXI(): Unit = runComponent("WEIGHT_AXI", new WEIGHT_AXI) { dut =>
    AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig()).reset()
    AxiMemorySim(dut.io.gmem2, dut.clockDomain, AxiMemorySimConfig()).reset()
    init_o_stream(dut.io.wq_stream)
    init_o_stream(dut.io.ws1_stream)
    init_o_stream(dut.io.ws2_stream)
    launchZeroLayer("WEIGHT_AXI", ctrl_cfg.IP_WEIGHT_AXI, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runKV_CACHE(): Unit = runComponent("KV_CACHE", new KV_CACHE) { dut =>
    AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig()).reset()
    init_o_stream(dut.io.kq_cache_i_stream)
    init_i_stream(dut.io.kq_cache_o_stream)
    init_o_stream(dut.io.ks_cache_i_stream)
    init_i_stream(dut.io.ks_cache_o_stream)
    init_o_stream(dut.io.vq_cache_i_stream)
    init_i_stream(dut.io.vq_cache_o_stream)
    init_o_stream(dut.io.vs_cache_i_stream)
    init_i_stream(dut.io.vs_cache_o_stream)
    launchZeroLayer("KV_CACHE", ctrl_cfg.IP_KV_CACHE, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runSTATE_AXI(): Unit = runComponent("STATE_AXI", new STATE_AXI) { dut =>
    AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig()).reset()
    init_o_stream(dut.io.x_stream)
    init_i_stream(dut.io.y_stream)
    init_i_stream(dut.io.cls_stream)
    launchZeroLayer("STATE_AXI", ctrl_cfg.IP_STATE_AXI, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runMUX(): Unit = runComponent("MUX", new MUX) { dut =>
    init_i_stream(dut.io.xlnq_stream)
    init_i_stream(dut.io.xlns_stream)
    init_i_stream(dut.io.aq_stream)
    init_i_stream(dut.io.as_stream)
    init_i_stream(dut.io.xmq_stream)
    init_i_stream(dut.io.xms_stream)
    init_o_stream(dut.io.q_stream)
    init_o_stream(dut.io.s_stream)
    launchZeroLayer("MUX", ctrl_cfg.IP_MUX, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runPERMUTE(): Unit = runComponent("PERMUTE", new PERMUTE) { dut =>
    init_i_stream(dut.io.i_stream)
    init_i_stream(dut.io.s_stream)
    init_i_stream(dut.io.w_stream)
    init_i_stream(dut.io.s1_stream)
    init_i_stream(dut.io.s2_stream)
    init_o_stream(dut.io.o_stream)
    launchZeroLayer("PERMUTE", ctrl_cfg.IP_PERMUTE, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runDEMUX(): Unit = runComponent("DEMUX", new DEMUX) { dut =>
    init_i_stream(dut.io.gemm_stream)
    init_i_stream(dut.io.bias_stream)
    init_o_stream(dut.io.qk_stream)
    init_o_stream(dut.io.v_stream)
    init_o_stream(dut.io.mlp1_stream)
    init_o_stream(dut.io.od_fc2_stream)
    init_o_stream(dut.io.cls_stream)
    launchZeroLayer("DEMUX", ctrl_cfg.IP_DEMUX, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runROPE_QK(): Unit = runComponent("ROPE_QK", new ROPE_QK) { dut =>
    init_i_stream(dut.io.qk_i_stream)
    init_o_stream(dut.io.qk_q_stream)
    init_o_stream(dut.io.qk_s_stream)
    launchZeroLayer("ROPE_QK", ctrl_cfg.IP_ROPE_QK, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runQK_GEMM(): Unit = runComponent("QK_GEMM", new QK_GEMM) { dut =>
    init_i_stream(dut.io.qk_q_stream)
    init_i_stream(dut.io.qk_s_stream)
    init_i_stream(dut.io.kq_cache_i_stream)
    init_i_stream(dut.io.ks_cache_i_stream)
    init_o_stream(dut.io.kq_cache_o_stream)
    init_o_stream(dut.io.ks_cache_o_stream)
    init_o_stream(dut.io.r_stream)
    launchZeroLayer("QK_GEMM", ctrl_cfg.IP_QK_GEMM, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runSOFTMAX(): Unit = runComponent("SOFTMAX", new SOFTMAX) { dut =>
    init_i_stream(dut.io.r_stream)
    init_o_stream(dut.io.rq_stream)
    init_o_stream(dut.io.rs_stream)
    launchZeroLayer("SOFTMAX", ctrl_cfg.IP_SOFTMAX, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runRV_GEMM(): Unit = runComponent("RV_GEMM", new RV_GEMM) { dut =>
    init_i_stream(dut.io.rq_stream)
    init_i_stream(dut.io.rs_stream)
    init_i_stream(dut.io.v_stream)
    init_i_stream(dut.io.vq_cache_i_stream)
    init_i_stream(dut.io.vs_cache_i_stream)
    init_o_stream(dut.io.vq_cache_o_stream)
    init_o_stream(dut.io.vs_cache_o_stream)
    init_o_stream(dut.io.aq_stream)
    init_o_stream(dut.io.as_stream)
    launchZeroLayer("RV_GEMM", ctrl_cfg.IP_RV_GEMM, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runA_REORDER(): Unit = runComponent("A_REORDER", new A_REORDER) { dut =>
    AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig()).reset()
    init_i_stream(dut.io.rv_aq_stream)
    init_i_stream(dut.io.rv_as_stream)
    init_o_stream(dut.io.mux_aq_stream)
    init_o_stream(dut.io.mux_as_stream)
    launchZeroLayer("A_REORDER", ctrl_cfg.IP_A_REORDER, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runSILU_GELU(): Unit = runComponent("SILU_GELU", new SILU_GELU) { dut =>
    init_i_stream(dut.io.mlp_i_stream)
    init_o_stream(dut.io.q_stream)
    init_o_stream(dut.io.s_stream)
    launchZeroLayer("SILU_GELU", ctrl_cfg.IP_SILU_GELU, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runXM_REORDER(): Unit = runComponent("XM_REORDER", new XM_REORDER) { dut =>
    AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig()).reset()
    init_i_stream(dut.io.silu_q_stream)
    init_i_stream(dut.io.silu_s_stream)
    init_o_stream(dut.io.mux_q_stream)
    init_o_stream(dut.io.mux_s_stream)
    launchZeroLayer("XM_REORDER", ctrl_cfg.IP_XM_REORDER, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runRESIDUAL(): Unit = runComponent("RESIDUAL", new RESIDUAL) { dut =>
    init_i_stream(dut.io.x_stream)
    init_i_stream(dut.io.res_i_stream)
    init_o_stream(dut.io.res_o_stream)
    init_o_stream(dut.io.y_stream)
    launchZeroLayer("RESIDUAL", ctrl_cfg.IP_RESIDUAL, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private def runRMS_LAYERNORM(): Unit = runComponent("RMS_LAYERNORM", new RMS_LAYERNORM) { dut =>
    init_i_stream(dut.io.x_stream)
    init_i_stream(dut.io.lnw_stream)
    init_i_stream(dut.io.lnb_stream)
    init_o_stream(dut.io.xlnq_stream)
    init_o_stream(dut.io.xlns_stream)
    launchZeroLayer("RMS_LAYERNORM", ctrl_cfg.IP_RMS_LAYERNORM, dut.io.signals, dut.io.idle, dut.clockDomain)
  }

  private val runners: Seq[(String, () => Unit)] = Seq(
    "M_AXI"          -> (() => runM_AXI()),
    "WEIGHT_AXI"     -> (() => runWEIGHT_AXI()),
    "KV_CACHE"       -> (() => runKV_CACHE()),
    "STATE_AXI"      -> (() => runSTATE_AXI()),
    "MUX"            -> (() => runMUX()),
    "PERMUTE"        -> (() => runPERMUTE()),
    "DEMUX"          -> (() => runDEMUX()),
    "ROPE_QK"        -> (() => runROPE_QK()),
    "QK_GEMM"        -> (() => runQK_GEMM()),
    "SOFTMAX"        -> (() => runSOFTMAX()),
    "RV_GEMM"        -> (() => runRV_GEMM()),
    "A_REORDER"      -> (() => runA_REORDER()),
    "SILU_GELU"      -> (() => runSILU_GELU()),
    "XM_REORDER"     -> (() => runXM_REORDER()),
    "RESIDUAL"       -> (() => runRESIDUAL()),
    "RMS_LAYERNORM"  -> (() => runRMS_LAYERNORM())
  )

  def run(args: Array[String]): Unit = {
    val selected =
      if (args.isEmpty) runners
      else {
        val wanted = args.map(_.toUpperCase).toSet
        runners.filter { case (name, _) => wanted.contains(name) }
      }
    require(selected.nonEmpty, s"No single-module simulation matched args: ${args.mkString(",")}")
    selected.foreach { case (_, runner) => runner() }
  }
}

object simulate_reuse_modules extends App { ReuseModuleSmokeSim.run(args) }

// Qwen 风格的单模块入口别名：需要只跑某个模块时可直接 Test/runMain 这些对象。
object simulate_m_axi extends App { ReuseModuleSmokeSim.run(Array("M_AXI")) }
object simulate_weight_axi extends App { ReuseModuleSmokeSim.run(Array("WEIGHT_AXI")) }
object simulate_kv_cache extends App { ReuseModuleSmokeSim.run(Array("KV_CACHE")) }
object simulate_state_axi extends App { ReuseModuleSmokeSim.run(Array("STATE_AXI")) }
object simulate_mux extends App { ReuseModuleSmokeSim.run(Array("MUX")) }
object simulate_permute extends App { ReuseModuleSmokeSim.run(Array("PERMUTE")) }
object simulate_demux extends App { ReuseModuleSmokeSim.run(Array("DEMUX")) }
object simulate_rope_qk extends App { ReuseModuleSmokeSim.run(Array("ROPE_QK")) }
object simulate_qk_gemm extends App { ReuseModuleSmokeSim.run(Array("QK_GEMM")) }
object simulate_softmax extends App { ReuseModuleSmokeSim.run(Array("SOFTMAX")) }
object simulate_rv_gemm extends App { ReuseModuleSmokeSim.run(Array("RV_GEMM")) }
object simulate_silu_gelu extends App { ReuseModuleSmokeSim.run(Array("SILU_GELU")) }
object simulate_residual extends App { ReuseModuleSmokeSim.run(Array("RESIDUAL")) }
object simulate_rms_layernorm extends App { ReuseModuleSmokeSim.run(Array("RMS_LAYERNORM")) }

// @formatter:on
