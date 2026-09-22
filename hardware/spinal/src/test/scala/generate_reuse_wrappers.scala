import spinal.core._

// @formatter:off
object generate_reuse_wrappers extends App {
  // 单模块封装 elaboration 测试入口：
  // 逐个实例化 wrapper，读取 src/main/verilog/<IP>/all.v 推导 AXIS/AXI 宽度，并生成 wrapper RTL。
  // 这一步不需要模型数据，主要用于捕获 Scala 黑盒端口名、宽度和 RTL 路径错误。

  val spinalConfig = SpinalConfig(
    defaultConfigForClockDomains = ClockDomainConfig(
      resetKind = ASYNC,
      resetActiveLevel = LOW
    ),
    mode = Verilog
  )

  val targets: Seq[(String, () => Component)] = Seq(
    "M_AXI"          -> (() => new M_AXI),
    "WEIGHT_AXI"     -> (() => new WEIGHT_AXI),
    "KV_CACHE"       -> (() => new KV_CACHE),
    "STATE_AXI"      -> (() => new STATE_AXI),
    "MUX"            -> (() => new MUX),
    "PERMUTE"        -> (() => new PERMUTE),
    "DEMUX"          -> (() => new DEMUX),
    "ROPE_QK"        -> (() => new ROPE_QK),
    "QK_GEMM"        -> (() => new QK_GEMM),
    "SOFTMAX"        -> (() => new SOFTMAX),
    "RV_GEMM"        -> (() => new RV_GEMM),
    "A_REORDER"      -> (() => new A_REORDER),
    "SILU_GELU"      -> (() => new SILU_GELU),
    "XM_REORDER"     -> (() => new XM_REORDER),
    "RESIDUAL"       -> (() => new RESIDUAL),
    "RMS_LAYERNORM"  -> (() => new RMS_LAYERNORM)
  )

  val selected = if (args.isEmpty) targets else {
    val wanted = args.map(_.toUpperCase).toSet
    targets.filter { case (name, _) => wanted.contains(name) }
  }

  require(selected.nonEmpty, s"No wrapper matched args: ${args.mkString(",")}")

  selected.foreach { case (name, gen) =>
    println(s"Generating wrapper for $name")
    SpinalVerilog(spinalConfig)(gen()).mergeRTLSource()
  }
}
