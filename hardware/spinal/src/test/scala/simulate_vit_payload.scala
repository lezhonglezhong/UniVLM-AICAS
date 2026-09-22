import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinal.lib.bus.amba4.axis.Axi4Stream.Axi4Stream
import utils._

import java.io.{BufferedInputStream, File, FileInputStream}
import scala.language.postfixOps

// @formatter:off

object VitPayloadSim {
  // ViT 真实 payload 单模块仿真。当前覆盖参数、权重和 memory-backed residual state：
  //   - PARAM_NORM: LayerNorm gamma/beta，layer -> pass -> CT -> CP。
  //   - PARAM_BIAS: DEMUX bias，按 H/QKV/TT/HCT/TP 和 TT/channel/TP 重放。
  //   - WEIGHT_AXI: DDR 中保存 compact 权重，mover 内按 TT 重放成 PERMUTE-facing WQ/WS 全量流。
  //   - STATE_AXI/RESIDUAL: token-major replay、ViT delta-order replay/writeback 和两次 residual pass。
  private val VIT_L = 12
  private val VIT_T = 1024
  private val VIT_TP = 8
  private val VIT_TT = VIT_T / VIT_TP
  private val VIT_C = 768
  private val VIT_CM = 3072
  private val VIT_H = 12
  private val VIT_HC = 64
  private val VIT_S = 1024
  private val VIT_ST = VIT_S / 8
  private val CP = 8
  private val VIT_CT = VIT_C / CP
  private val VIT_CMT = VIT_CM / CP
  private val VIT_HCT = VIT_HC / CP
  private val VIT_NUM_X = VIT_T * VIT_C
  private val VIT_NUM_X2 = 2 * VIT_NUM_X
  private val VIT_NUM_QK = 2 * VIT_T * VIT_C
  private val VIT_NUM_V = VIT_T * VIT_C
  private val VIT_NUM_FC1 = VIT_T * VIT_CM
  private val VIT_NUM_R = VIT_H * VIT_T * VIT_S
  private val VIT_NUM_RS = VIT_H * VIT_T * VIT_ST
  private val VIT_NUM_AQ = VIT_H * VIT_T * VIT_HC
  private val VIT_NUM_AS = VIT_H * VIT_T * VIT_HCT
  private val VIT_LAYER_BIAS = 3 * VIT_C + VIT_C + VIT_CM + VIT_C
  private val VIT_QKV_W = VIT_HC * VIT_C
  private val VIT_QKV_WS = VIT_QKV_W / CP
  private val VIT_O_W = VIT_C * VIT_C
  private val VIT_O_WS = VIT_O_W / CP
  private val VIT_FC1_W = VIT_CM * VIT_C
  private val VIT_FC1_WS = VIT_FC1_W / CP
  private val VIT_FC2_W = VIT_C * VIT_CM
  private val VIT_FC2_WS = VIT_FC2_W / CP
  private val VIT_COMPACT_WQ = 4 * VIT_C * VIT_C + 2 * VIT_C * VIT_CM
  private val VIT_COMPACT_WS = VIT_COMPACT_WQ / CP
  private val VIT_BIAS_REPLAY =
    VIT_H * 3 * VIT_TT * VIT_HCT * VIT_TP * CP +
    VIT_TT * VIT_CT * VIT_TP * CP +
    VIT_TT * VIT_CMT * VIT_TP * CP +
    VIT_TT * VIT_CT * VIT_TP * CP

  private val packedPath = sys.env.getOrElse("VLM_PACKED_WEIGHT_PATH", ctrl_cfg.packed_path)
  private val vitBinaryPrefix = sys.env.getOrElse("VLM_VIT_BINARIES_PREFIX", s"${ctrl_cfg.weights_root}/ViT/binaries/vision_")
  private val timeoutCycles = sys.env.getOrElse("VLM_VIT_PAYLOAD_TIMEOUT", "200000000").toInt
  private val weightPrefixTt = sys.env.getOrElse("VLM_VIT_WEIGHT_PREFIX_TT", "3").toInt

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
    if (sys.env.get("VLM_VIT_PAYLOAD_WAVE").contains("1")) base.withFstWave else base
  }

  sealed trait ElemKind { def bytes: Int }
  case object I8 extends ElemKind { val bytes = 1 }
  case object I32 extends ElemKind { val bytes = 4 }
  case object I64 extends ElemKind { val bytes = 8 }

  private def vitLayerDir(l: Int): String = s"${ctrl_cfg.vit_condense_prefix}$l"
  private def vitBinaryDir(l: Int): String = s"$vitBinaryPrefix$l"
  private def packed(name: String): String = s"$packedPath/$name"

  private def selectedLayers: Seq[Int] =
    sys.env.getOrElse("VLM_VIT_TEST_LAYERS", "0").split("[,\\s]+").filter(_.nonEmpty).map(_.toInt).toSeq

  private def fileElemCount(path: String, kind: ElemKind): Int = {
    val f = new File(path)
    require(f.isFile, s"Missing payload file: $path")
    require(f.length() % kind.bytes == 0, s"File size is not aligned: $path")
    (f.length() / kind.bytes).toInt
  }

  private def signExtend(v: BigInt, bits: Int): Long = {
    val sign = BigInt(1) << (bits - 1)
    val mod = BigInt(1) << bits
    val masked = v & (mod - 1)
    (if ((masked & sign) != 0) masked - mod else masked).toLong
  }

  private def normalizedExpected(v: Long, laneBits: Int, signed: Boolean): Long =
    if (signed) signExtend(BigInt(v), laneBits) else (BigInt(v) & ((BigInt(1) << laneBits) - 1)).toLong

  private def readValues(path: String, kind: ElemKind): Array[Long] = kind match {
    case I8  => read_int8_file(path, fileElemCount(path, kind)).map(_.toLong)
    case I64 => read_int64_file(path, fileElemCount(path, kind))
    case I32 =>
      val bytes = read_int8_file(path, fileElemCount(path, kind) * kind.bytes)
      val out = Array.ofDim[Long](bytes.length / 4)
      for (i <- out.indices) {
        val raw =
          (BigInt(bytes(i * 4 + 0) & 0xff) << 0) |
          (BigInt(bytes(i * 4 + 1) & 0xff) << 8) |
          (BigInt(bytes(i * 4 + 2) & 0xff) << 16) |
          (BigInt(bytes(i * 4 + 3) & 0xff) << 24)
        out(i) = signExtend(raw, 32)
      }
      out
  }

  private def writeBytes(mem: AxiMemorySim, base: Long, bytes: Array[Byte]): Unit =
    for (i <- bytes.indices) mem.memory.write(base + i, bytes(i))

  private def writeI32Values(mem: AxiMemorySim, base: Long, values: Array[Long]): Unit = {
    for (i <- values.indices) {
      val raw = BigInt(values(i)) & 0xffffffffL
      mem.memory.writeBigInt(base + i * 4L, raw, 4)
    }
  }

  private def readI32Values(mem: AxiMemorySim, base: Long, count: Int): Array[Long] = {
    val out = Array.ofDim[Long](count)
    for (i <- 0 until count) out(i) = signExtend(mem.memory.readBigInt(base + i * 4L, 4), 32)
    out
  }

  private def splitPackedToMem(memLo: AxiMemorySim, memHi: AxiMemorySim, baseLo: Long, baseHi: Long, path: String): Unit = {
    // packed ViT 权重文件采用 all_layer_lo_half ++ all_layer_hi_half；两路 AXI 从各自 base 读取同层半区。
    val bytes = read_int8_file(path, fileElemCount(path, I8))
    require(bytes.length % 2 == 0, s"Packed weight file must have even length: $path")
    val half = bytes.length / 2
    writeBytes(memLo, baseLo, bytes.slice(0, half))
    writeBytes(memHi, baseHi, bytes.slice(half, bytes.length))
  }

  private def expectValues(
                            stream: Axi4Stream,
                            clockDomain: ClockDomain,
                            expected: Array[Long],
                            lanes: Int,
                            signed: Boolean,
                            info: String
                          ): Unit = {
    require(expected.length % lanes == 0, s"$info length ${expected.length} is not divisible by lanes=$lanes")
    val laneBits = stream.config.dataWidth * 8 / lanes
    var mismatch = 0
    var index = 0
    stream.ready #= true
    while (index < expected.length) {
      clockDomain.waitSamplingWhere(stream.valid.toBoolean && stream.ready.toBoolean)
      val tile = stream.data.toBigInt
      for (lane <- 0 until lanes) {
        val gotBits = (tile >> (laneBits * lane)) & ((BigInt(1) << laneBits) - 1)
        val got = if (signed) signExtend(gotBits, laneBits) else gotBits.toLong
        val exp = normalizedExpected(expected(index + lane), laneBits, signed)
        if (got != exp) {
          mismatch += 1
          if (mismatch <= 10) println(s"$info mismatch at ${index + lane}: got=$got expected=$exp laneBits=$laneBits")
        }
      }
      index += lanes
    }
    stream.ready #= false
    require(mismatch == 0, s"$info has $mismatch mismatches")
    println(s"output $info matched (${expected.length} elements, $lanes lanes)")
  }

  private def feedValues(
                          stream: Axi4Stream,
                          clockDomain: ClockDomain,
                          values: Array[Long],
                          lanes: Int,
                          info: String
                        ): Unit = {
    require(values.length % lanes == 0, s"$info length ${values.length} is not divisible by lanes=$lanes")
    val laneBits = stream.config.dataWidth * 8 / lanes
    for (base <- values.indices by lanes) {
      val tile = values.slice(base, base + lanes)
      stream.valid #= true
      stream.data #= compose_tile(tile, laneBits)
      clockDomain.waitSamplingWhere(stream.valid.toBoolean && stream.ready.toBoolean)
      stream.valid #= false
    }
    println(s"input $info done (${values.length} elements, $lanes lanes)")
  }

  private def feedFile(
                        stream: Axi4Stream,
                        clockDomain: ClockDomain,
                        path: String,
                        kind: ElemKind,
                        lanes: Int,
                        info: String
                      ): Unit = feedValues(stream, clockDomain, readValues(path, kind), lanes, info)

  private def expectFile(
                          stream: Axi4Stream,
                          clockDomain: ClockDomain,
                          path: String,
                          kind: ElemKind,
                          lanes: Int,
                          signed: Boolean,
                          info: String
                        ): Unit = expectValues(stream, clockDomain, readValues(path, kind), lanes, signed, info)

  private def expectValuesWithMarker(
                                      stream: Axi4Stream,
                                      clockDomain: ClockDomain,
                                      expected: Array[Long],
                                      lanes: Int,
                                      signed: Boolean,
                                      info: String,
                                      markerElements: Int,
                                      mark: () => Unit
                                    ): Unit = {
    // STATE_AXI layer 闭环测试必须先观察 delta-order x 发出，再喂 y，避免提前供数掩盖 x/y 互等。
    require(expected.length % lanes == 0, s"$info length ${expected.length} is not divisible by lanes=$lanes")
    require(markerElements >= 0 && markerElements <= expected.length, s"$info marker $markerElements out of range")
    val laneBits = stream.config.dataWidth * 8 / lanes
    var mismatch = 0
    var index = 0
    var marked = false
    stream.ready #= true
    while (index < expected.length) {
      clockDomain.waitSamplingWhere(stream.valid.toBoolean && stream.ready.toBoolean)
      val tile = stream.data.toBigInt
      for (lane <- 0 until lanes) {
        val gotBits = (tile >> (laneBits * lane)) & ((BigInt(1) << laneBits) - 1)
        val got = if (signed) signExtend(gotBits, laneBits) else gotBits.toLong
        val exp = normalizedExpected(expected(index + lane), laneBits, signed)
        if (got != exp) {
          mismatch += 1
          if (mismatch <= 10) println(s"$info mismatch at ${index + lane}: got=$got expected=$exp laneBits=$laneBits")
        }
      }
      index += lanes
      if (!marked && index >= markerElements) {
        marked = true
        mark()
      }
    }
    stream.ready #= false
    require(mismatch == 0, s"$info has $mismatch mismatches")
    println(s"output $info matched (${expected.length} elements, $lanes lanes)")
  }

  private def expectI8FileStreaming(
                                      stream: Axi4Stream,
                                      clockDomain: ClockDomain,
                                      path: String,
                                      lanes: Int,
                                      signed: Boolean,
                                      info: String
                                    ): Unit = {
    // ViT WEIGHT_AXI 的 WQ full replay 接近 1GB，必须流式读参考文件，避免 Scala 测试先展开成 Long 数组。
    val expectedLength = fileElemCount(path, I8)
    require(expectedLength % lanes == 0, s"$info length $expectedLength is not divisible by lanes=$lanes")
    val laneBits = stream.config.dataWidth * 8 / lanes
    val in = new BufferedInputStream(new FileInputStream(path), 1 << 20)
    var mismatch = 0
    var index = 0
    stream.ready #= true
    try {
      while (index < expectedLength) {
        clockDomain.waitSamplingWhere(stream.valid.toBoolean && stream.ready.toBoolean)
        val tile = stream.data.toBigInt
        for (lane <- 0 until lanes) {
          val raw = in.read()
          require(raw >= 0, s"$info ended early at element ${index + lane}")
          val expByte = if (raw >= 128) raw - 256 else raw
          val gotBits = (tile >> (laneBits * lane)) & ((BigInt(1) << laneBits) - 1)
          val got = if (signed) signExtend(gotBits, laneBits) else gotBits.toLong
          val exp = normalizedExpected(expByte.toLong, laneBits, signed)
          if (got != exp) {
            mismatch += 1
            if (mismatch <= 10) println(s"$info mismatch at ${index + lane}: got=$got expected=$exp laneBits=$laneBits")
          }
        }
        index += lanes
      }
      require(in.read() < 0, s"$info reference file has trailing data")
    } finally {
      in.close()
      stream.ready #= false
    }
    require(mismatch == 0, s"$info has $mismatch mismatches")
    println(s"output $info matched ($expectedLength elements, $lanes lanes)")
  }

  private def feedI8FileStreaming(
                                   stream: Axi4Stream,
                                   clockDomain: ClockDomain,
                                   path: String,
                                   lanes: Int,
                                   signed: Boolean,
                                   info: String,
                                   elements: Int = -1
                                 ): Unit = {
    // PERMUTE/MUX 的 ViT GEMM-facing 文件接近 GB 级，流式喂入避免测试端持有大数组。
    val total = fileElemCount(path, I8)
    val count = if (elements < 0) total else elements
    require(count > 0 && count <= total, s"$info element count $count out of file range $total")
    require(count % lanes == 0, s"$info length $count is not divisible by lanes=$lanes")
    val laneBits = stream.config.dataWidth * 8 / lanes
    val in = new BufferedInputStream(new FileInputStream(path), 1 << 20)
    var index = 0
    try {
      while (index < count) {
        val tile = Array.ofDim[Long](lanes)
        for (lane <- 0 until lanes) {
          val raw = in.read()
          require(raw >= 0, s"$info ended early at element ${index + lane}")
          tile(lane) = if (signed && raw >= 128) raw - 256 else raw
        }
        stream.valid #= true
        stream.data #= compose_tile(tile, laneBits)
        clockDomain.waitSamplingWhere(stream.valid.toBoolean && stream.ready.toBoolean)
        stream.valid #= false
        index += lanes
      }
    } finally {
      in.close()
      stream.valid #= false
    }
    println(s"input $info done ($count elements, $lanes lanes)")
  }

  private def expectI8FilePrefix(
                                  stream: Axi4Stream,
                                  clockDomain: ClockDomain,
                                  path: String,
                                  elements: Int,
                                  lanes: Int,
                                  signed: Boolean,
                                  info: String
                                ): Unit = {
    // Quick payload：只验证多个 TT block 和边界顺序，然后直接结束仿真，避免为单模块迭代 drain 完整 layer。
    require(elements > 0 && elements % lanes == 0, s"$info prefix length $elements is not divisible by lanes=$lanes")
    require(fileElemCount(path, I8) >= elements, s"$info prefix $elements exceeds file length")
    val laneBits = stream.config.dataWidth * 8 / lanes
    val in = new BufferedInputStream(new FileInputStream(path), 1 << 20)
    var mismatch = 0
    var index = 0
    stream.ready #= true
    try {
      while (index < elements) {
        clockDomain.waitSamplingWhere(stream.valid.toBoolean && stream.ready.toBoolean)
        val tile = stream.data.toBigInt
        for (lane <- 0 until lanes) {
          val raw = in.read()
          require(raw >= 0, s"$info ended early at element ${index + lane}")
          val expByte = if (raw >= 128) raw - 256 else raw
          val gotBits = (tile >> (laneBits * lane)) & ((BigInt(1) << laneBits) - 1)
          val got = if (signed) signExtend(gotBits, laneBits) else gotBits.toLong
          val exp = normalizedExpected(expByte.toLong, laneBits, signed)
          if (got != exp) {
            mismatch += 1
            if (mismatch <= 10) println(s"$info mismatch at ${index + lane}: got=$got expected=$exp laneBits=$laneBits")
          }
        }
        index += lanes
      }
    } finally {
      in.close()
      stream.ready #= false
    }
    require(mismatch == 0, s"$info has $mismatch mismatches in prefix")
    println(s"output $info prefix matched ($elements elements, $lanes lanes)")
  }

  private def initAndLaunch(
                             signals: DaisyChain[ManagerSignals],
                             clockDomain: ClockDomain,
                             lBegin: Int,
                             lClose: Int,
                             paramOp: Int,
                             ipBit: Int = ctrl_cfg.IP_M_AXI,
                             stateOp: Int = 0
                           ): Unit = {
    signals.I.MODE #= 1
    signals.I.L_BEGIN #= lBegin
    signals.I.L_CLOSE #= lClose
    signals.I.PARAM_OP #= paramOp
    signals.I.STATE_OP #= stateOp
    signals.I.RUN_MASK #= (BigInt(1) << ipBit)
    signals.I.CACHE_UPDATE_MODE #= 0
    clockDomain.waitSampling(20)
    signals.I.T #= true
    clockDomain.waitSampling()
    signals.I.T #= false
  }

  private def startDut(signals: DaisyChain[ManagerSignals], clockDomain: ClockDomain): Unit = {
    init_daisy_chain(signals)
    init_clock(clockDomain, 10)
  }

  private def waitIdle(clockDomain: ClockDomain, idle: Bool, name: String): Unit = {
    var cycles = 0
    while (idle.toBoolean && cycles < timeoutCycles) {
      clockDomain.waitSampling()
      cycles += 1
    }
    require(!idle.toBoolean, s"$name did not leave idle within $timeoutCycles cycles")
    cycles = 0
    while (!idle.toBoolean && cycles < timeoutCycles) {
      clockDomain.waitSampling()
      cycles += 1
    }
    require(idle.toBoolean, s"$name did not return idle within $timeoutCycles cycles")
  }

  private def expectedBiasReplay(layerBias: Array[Long]): Array[Long] = {
    require(layerBias.length == VIT_LAYER_BIAS, s"Unexpected layer bias length ${layerBias.length}")
    val out = Array.ofDim[Long](VIT_BIAS_REPLAY)
    var idx = 0
    val offBq = 0
    val offBk = offBq + VIT_C
    val offBv = offBk + VIT_C
    val offBo = offBv + VIT_C
    val offB1 = offBo + VIT_C
    val offB2 = offB1 + VIT_CM

    for (h <- 0 until VIT_H; qkv <- 0 until 3; tt <- 0 until VIT_TT; hct <- 0 until VIT_HCT; tp <- 0 until VIT_TP; cp <- 0 until CP) {
      val off = if (qkv == 0) offBq else if (qkv == 1) offBk else offBv
      out(idx) = layerBias(off + h * VIT_HC + hct * CP + cp)
      idx += 1
    }
    for (tt <- 0 until VIT_TT; ct <- 0 until VIT_CT; tp <- 0 until VIT_TP; cp <- 0 until CP) {
      out(idx) = layerBias(offBo + ct * CP + cp)
      idx += 1
    }
    for (tt <- 0 until VIT_TT; cmt <- 0 until VIT_CMT; tp <- 0 until VIT_TP; cp <- 0 until CP) {
      out(idx) = layerBias(offB1 + cmt * CP + cp)
      idx += 1
    }
    for (tt <- 0 until VIT_TT; ct <- 0 until VIT_CT; tp <- 0 until VIT_TP; cp <- 0 until CP) {
      out(idx) = layerBias(offB2 + ct * CP + cp)
      idx += 1
    }
    require(idx == out.length, s"Bias replay generated $idx, expected ${out.length}")
    out
  }

  private def vitTokenWindow(path: String): Array[Long] = {
    // ViT binary 激活文件按 T_LOAD=13312 保存；当前硬件 layer payload 只取前 1024 token。
    val all = readValues(path, I64)
    require(all.length % VIT_C == 0, s"$path element count ${all.length} is not divisible by VIT_C=$VIT_C")
    require(all.length >= VIT_NUM_X, s"$path has ${all.length} elements, smaller than ViT layer state $VIT_NUM_X")
    all.slice(0, VIT_NUM_X)
  }

  private def vitDeltaOrder(values: Array[Long]): Array[Long] = {
    // ViT RESIDUAL delta/writeback 顺序为 TT_D -> CT -> TP -> CP，memory/state 文件本身是 token-major。
    require(values.length == VIT_NUM_X, s"ViT delta-order source length ${values.length} != $VIT_NUM_X")
    val out = Array.ofDim[Long](VIT_NUM_X)
    var idx = 0
    for (tt <- 0 until VIT_TT; ct <- 0 until VIT_CT; tp <- 0 until VIT_TP; cp <- 0 until CP) {
      val token = tt * VIT_TP + tp
      out(idx) = values(token * VIT_C + ct * CP + cp)
      idx += 1
    }
    out
  }

  private def vitAStreamOrder(values: Array[Long]): Array[Long] = {
    // RV_GEMM condensed A 是 H -> token -> HCT -> CP；MUX 需要 TT -> H -> TP -> HCT -> CP。
    require(values.length == VIT_NUM_AQ, s"ViT A_Q length ${values.length} != $VIT_NUM_AQ")
    val out = Array.ofDim[Long](VIT_NUM_AQ)
    var idx = 0
    for (tt <- 0 until VIT_TT; h <- 0 until VIT_H; tp <- 0 until VIT_TP; hct <- 0 until VIT_HCT; cp <- 0 until CP) {
      val token = tt * VIT_TP + tp
      out(idx) = values(h * VIT_T * VIT_HC + token * VIT_HC + hct * CP + cp)
      idx += 1
    }
    out
  }

  private def vitAScaleStreamOrder(values: Array[Long]): Array[Long] = {
    // A scale 同步重排为 TT -> H -> TP -> HCT。
    require(values.length == VIT_NUM_AS, s"ViT A_S length ${values.length} != $VIT_NUM_AS")
    val out = Array.ofDim[Long](VIT_NUM_AS)
    var idx = 0
    for (tt <- 0 until VIT_TT; h <- 0 until VIT_H; tp <- 0 until VIT_TP; hct <- 0 until VIT_HCT) {
      val token = tt * VIT_TP + tp
      out(idx) = values(h * VIT_T * VIT_HCT + token * VIT_HCT + hct)
      idx += 1
    }
    out
  }

  private def vitNormParams(l: Int): (Array[Long], Array[Long]) = {
    val allLnw = readValues(packed("all_vit_lnw_i32.bin"), I32)
    val allLnb = readValues(packed("all_vit_lnb_i64.bin"), I64)
    val start = l * 2 * VIT_C
    (allLnw.slice(start, start + 2 * VIT_C), allLnb.slice(start, start + 2 * VIT_C))
  }

  def runM_AXI(): Unit = {
    simConfig.compile(new M_AXI).doSimUntilVoid { dut =>
      val axi = AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig())
      axi.reset()
      init_o_stream(dut.io.bias_stream)
      init_o_stream(dut.io.lnw_stream)
      init_o_stream(dut.io.lnb_stream)
      startDut(dut.io.signals, dut.clockDomain)

      val biasBase = 0x1000L
      val lnwBase = 0x100000L
      val lnbBase = 0x200000L
      writeBytes(axi, biasBase, read_int8_file(packed("all_vit_bias_i32.bin"), fileElemCount(packed("all_vit_bias_i32.bin"), I8)))
      writeBytes(axi, lnwBase, read_int8_file(packed("all_vit_lnw_i32.bin"), fileElemCount(packed("all_vit_lnw_i32.bin"), I8)))
      writeBytes(axi, lnbBase, read_int8_file(packed("all_vit_lnb_i64.bin"), fileElemCount(packed("all_vit_lnb_i64.bin"), I8)))

      dut.io.signals.I.MEMORY_VIT_BIAS #= biasBase
      dut.io.signals.I.MEMORY_VIT_LNW #= lnwBase
      dut.io.signals.I.MEMORY_VIT_LNB #= lnbBase

      val allBias = readValues(packed("all_vit_bias_i32.bin"), I32)
      val allLnw = readValues(packed("all_vit_lnw_i32.bin"), I32)
      val allLnb = readValues(packed("all_vit_lnb_i64.bin"), I64)

      selectedLayers.foreach { l =>
        val lnStart = l * 2 * VIT_C
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 1) },
          fork { expectValues(dut.io.lnw_stream, dut.clockDomain, allLnw.slice(lnStart, lnStart + 2 * VIT_C), CP, signed = true, s"M_AXI ViT lnw layer $l") },
          fork { expectValues(dut.io.lnb_stream, dut.clockDomain, allLnb.slice(lnStart, lnStart + 2 * VIT_C), CP, signed = true, s"M_AXI ViT lnb layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"M_AXI ViT norm layer $l") }
        ).foreach(_.join())

        val biasStart = l * VIT_LAYER_BIAS
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0) },
          fork { expectValues(dut.io.bias_stream, dut.clockDomain, expectedBiasReplay(allBias.slice(biasStart, biasStart + VIT_LAYER_BIAS)), CP, signed = true, s"M_AXI ViT bias layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"M_AXI ViT bias layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  def runWEIGHT_AXI(): Unit = {
    simConfig.compile(new WEIGHT_AXI).doSimUntilVoid { dut =>
      val axiLo = AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig())
      val axiHi = AxiMemorySim(dut.io.gmem2, dut.clockDomain, AxiMemorySimConfig())
      axiLo.reset()
      axiHi.reset()
      init_o_stream(dut.io.wq_stream)
      init_o_stream(dut.io.ws1_stream)
      init_o_stream(dut.io.ws2_stream)
      startDut(dut.io.signals, dut.clockDomain)

      val vitBase = 0x1000L
      splitPackedToMem(axiLo, axiHi, vitBase, vitBase, packed("all_vit_w.bin"))
      dut.io.signals.I.MEMORY_VIT_W_LO #= vitBase
      dut.io.signals.I.MEMORY_VIT_W_HI #= vitBase

      selectedLayers.foreach { l =>
        val dir = vitLayerDir(l)
        val qkvPrefixWq = VIT_QKV_W * (weightPrefixTt + 1)
        val qkvPrefixWs = VIT_QKV_WS * (weightPrefixTt + 1)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_WEIGHT_AXI) },
          fork { expectI8FilePrefix(dut.io.wq_stream,  dut.clockDomain, s"$dir/CONDENSED_GEMM_W_Q.bin",  qkvPrefixWq, 64, signed = true,  s"WEIGHT_AXI ViT WQ layer $l") },
          fork { expectI8FilePrefix(dut.io.ws1_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_W_S1.bin", qkvPrefixWs, 8,  signed = false, s"WEIGHT_AXI ViT WS1 layer $l") },
          fork { expectI8FilePrefix(dut.io.ws2_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_W_S2.bin", qkvPrefixWs, 8,  signed = false, s"WEIGHT_AXI ViT WS2 layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  def runSTATE_AXI(): Unit = {
    simConfig.compile(new STATE_AXI).doSimUntilVoid { dut =>
      val axi = AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig())
      axi.reset()
      init_o_stream(dut.io.x_stream)
      init_i_stream(dut.io.y_stream)
      init_i_stream(dut.io.cls_stream)
      startDut(dut.io.signals, dut.clockDomain)

      val stateBase = 0x1000L
      dut.io.signals.I.MEMORY_VIT_STATE #= stateBase

      selectedLayers.foreach { l =>
        val bin = vitBinaryDir(l)
        val mhaX = vitTokenWindow(s"$bin/MHA_LN_X.bin")
        val mhaORes = vitTokenWindow(s"$bin/MHA_O_RES.bin")
        val mlpX = vitTokenWindow(s"$bin/MLP_LN_X.bin")
        val mlpXfc2Res = vitTokenWindow(s"$bin/MLP_XFC2_RES.bin")
        require(mhaORes.sameElements(mlpX), s"ViT layer $l MHA_O_RES and MLP_LN_X differ; pass1 state reference is ambiguous")

        writeI32Values(axi, stateBase, mhaX)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_STATE_AXI, stateOp = 0) },
          fork { expectValues(dut.io.x_stream, dut.clockDomain, mhaX, CP, signed = true, s"STATE_AXI ViT token replay layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"STATE_AXI ViT token layer $l") }
        ).foreach(_.join())

        writeI32Values(axi, stateBase, mhaX)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_STATE_AXI, stateOp = 1) },
          fork { expectValues(dut.io.x_stream, dut.clockDomain, vitDeltaOrder(mhaX), CP, signed = true, s"STATE_AXI ViT delta replay layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"STATE_AXI ViT delta layer $l") }
        ).foreach(_.join())

        writeI32Values(axi, stateBase, mhaX)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_STATE_AXI, stateOp = 2) },
          fork { feedValues(dut.io.y_stream, dut.clockDomain, vitDeltaOrder(mhaORes), CP, s"STATE_AXI ViT writeback input layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"STATE_AXI ViT writeback layer $l") }
        ).foreach(_.join())
        require(readI32Values(axi, stateBase, VIT_NUM_X).sameElements(mhaORes), s"STATE_AXI ViT writeback memory mismatch for layer $l")

        val layerXExpected = mhaX ++ vitDeltaOrder(mhaX) ++ mhaORes ++ vitDeltaOrder(mhaORes)
        val layerYPayload = vitDeltaOrder(mhaORes) ++ vitDeltaOrder(mlpXfc2Res)
        @volatile var sawDeltaReplay = false
        writeI32Values(axi, stateBase, mhaX)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_STATE_AXI, stateOp = 4) },
          fork { expectValuesWithMarker(dut.io.x_stream, dut.clockDomain, layerXExpected, CP, signed = true, s"STATE_AXI ViT layer x replay $l", markerElements = VIT_NUM_X + CP, mark = () => sawDeltaReplay = true) },
          fork {
            var cycles = 0
            while (!sawDeltaReplay && cycles < timeoutCycles) {
              dut.clockDomain.waitSampling()
              cycles += 1
            }
            require(sawDeltaReplay, s"STATE_AXI ViT layer $l did not emit delta-order x before y feed")
            feedValues(dut.io.y_stream, dut.clockDomain, layerYPayload, CP, s"STATE_AXI ViT layer y writeback input $l")
          },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"STATE_AXI ViT layer op $l") }
        ).foreach(_.join())
        require(readI32Values(axi, stateBase, VIT_NUM_X).sameElements(mlpXfc2Res), s"STATE_AXI ViT layer memory mismatch for layer $l")
      }
      simSuccess()
    }
  }

  def runMUX(): Unit = {
    simConfig.compile(new MUX).doSimUntilVoid { dut =>
      init_i_stream(dut.io.xlnq_stream); init_i_stream(dut.io.xlns_stream)
      init_i_stream(dut.io.aq_stream);   init_i_stream(dut.io.as_stream)
      init_i_stream(dut.io.xmq_stream);  init_i_stream(dut.io.xms_stream)
      init_o_stream(dut.io.q_stream);    init_o_stream(dut.io.s_stream)
      startDut(dut.io.signals, dut.clockDomain)

      selectedLayers.foreach { l =>
        val dir = vitLayerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_MUX) },
          fork { feedFile(dut.io.xlnq_stream, dut.clockDomain, s"$dir/CONDENSED_XLN_Q.bin", I8, 8, "MUX ViT XLN_Q") },
          fork { feedFile(dut.io.xlns_stream, dut.clockDomain, s"$dir/CONDENSED_XLN_S.bin", I8, 1, "MUX ViT XLN_S") },
          fork { feedValues(dut.io.aq_stream, dut.clockDomain, vitAStreamOrder(readValues(s"$dir/CONDENSED_RV_GEMM_A_Q.bin", I8)), 8, "MUX ViT A_Q") },
          fork { feedValues(dut.io.as_stream, dut.clockDomain, vitAScaleStreamOrder(readValues(s"$dir/CONDENSED_RV_GEMM_A_S.bin", I8)), 1, "MUX ViT A_S") },
          fork { feedFile(dut.io.xmq_stream, dut.clockDomain, s"$dir/CONDENSED_GELU_XM_Q.bin", I8, 8, "MUX ViT XM_Q") },
          fork { feedFile(dut.io.xms_stream, dut.clockDomain, s"$dir/CONDENSED_GELU_XM_S.bin", I8, 1, "MUX ViT XM_S") },
          fork { expectI8FileStreaming(dut.io.q_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_X_Q.bin", 64, signed = true,  s"MUX ViT X_Q layer $l") },
          fork { expectI8FileStreaming(dut.io.s_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_X_S.bin", 8,  signed = false, s"MUX ViT X_S layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"MUX ViT layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  def runPERMUTE(): Unit = {
    simConfig.compile(new PERMUTE).doSimUntilVoid { dut =>
      init_i_stream(dut.io.i_stream); init_i_stream(dut.io.s_stream); init_i_stream(dut.io.w_stream)
      init_i_stream(dut.io.s1_stream); init_i_stream(dut.io.s2_stream); init_o_stream(dut.io.o_stream)
      startDut(dut.io.signals, dut.clockDomain)

      selectedLayers.foreach { l =>
        val dir = vitLayerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_PERMUTE) },
          fork { feedI8FileStreaming(dut.io.i_stream,  dut.clockDomain, s"$dir/CONDENSED_GEMM_X_Q.bin",  64, signed = true,  s"PERMUTE ViT X_Q layer $l") },
          fork { feedI8FileStreaming(dut.io.s_stream,  dut.clockDomain, s"$dir/CONDENSED_GEMM_X_S.bin",  8,  signed = false, s"PERMUTE ViT X_S layer $l") },
          fork { feedI8FileStreaming(dut.io.w_stream,  dut.clockDomain, s"$dir/CONDENSED_GEMM_W_Q.bin",  64, signed = true,  s"PERMUTE ViT W_Q layer $l") },
          fork { feedI8FileStreaming(dut.io.s1_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_W_S1.bin", 8,  signed = false, s"PERMUTE ViT W_S1 layer $l") },
          fork { feedI8FileStreaming(dut.io.s2_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_W_S2.bin", 8,  signed = false, s"PERMUTE ViT W_S2 layer $l") },
          fork { expectFile(dut.io.o_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_Y.bin", I64, 8, signed = true, s"PERMUTE ViT Y layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"PERMUTE ViT layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  def runDEMUX(): Unit = {
    simConfig.compile(new DEMUX).doSimUntilVoid { dut =>
      init_i_stream(dut.io.gemm_stream); init_i_stream(dut.io.bias_stream)
      init_o_stream(dut.io.qk_stream); init_o_stream(dut.io.v_stream); init_o_stream(dut.io.mlp1_stream)
      init_o_stream(dut.io.od_fc2_stream); init_o_stream(dut.io.cls_stream)
      startDut(dut.io.signals, dut.clockDomain)
      val allBias = readValues(packed("all_vit_bias_i32.bin"), I32)

      selectedLayers.foreach { l =>
        val dir = vitLayerDir(l)
        val biasStart = l * VIT_LAYER_BIAS
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_DEMUX) },
          fork { feedFile(dut.io.gemm_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_Y.bin", I64, 8, "DEMUX ViT GEMM_Y") },
          fork { feedValues(dut.io.bias_stream, dut.clockDomain, expectedBiasReplay(allBias.slice(biasStart, biasStart + VIT_LAYER_BIAS)), CP, "DEMUX ViT bias") },
          fork { expectFile(dut.io.qk_stream,     dut.clockDomain, s"$dir/CONDENSED_DEMUX_QK.bin",   I64, 8, signed = true, s"DEMUX ViT QK layer $l") },
          fork { expectFile(dut.io.v_stream,      dut.clockDomain, s"$dir/CONDENSED_DEMUX_V.bin",    I64, 8, signed = true, s"DEMUX ViT V layer $l") },
          fork { expectFile(dut.io.mlp1_stream,   dut.clockDomain, s"$dir/CONDENSED_DEMUX_FC1.bin",  I64, 8, signed = true, s"DEMUX ViT FC1 layer $l") },
          fork { expectFile(dut.io.od_fc2_stream, dut.clockDomain, s"$dir/CONDENSED_DEMUX_OFC2.bin", I64, 8, signed = true, s"DEMUX ViT OFC2 layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"DEMUX ViT layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  def runROPE_QK(): Unit = {
    simConfig.compile(new ROPE_QK).doSimUntilVoid { dut =>
      init_i_stream(dut.io.qk_i_stream); init_o_stream(dut.io.qk_q_stream); init_o_stream(dut.io.qk_s_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = vitLayerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_ROPE_QK) },
          fork { feedFile(dut.io.qk_i_stream, dut.clockDomain, s"$dir/CONDENSED_DEMUX_QK.bin", I64, 8, "ROPE_QK ViT input") },
          fork { expectFile(dut.io.qk_q_stream, dut.clockDomain, s"$dir/CONDENSED_ROPE_QK_BYPASS_Q.bin", I8, 8, signed = true,  s"ROPE_QK ViT Q layer $l") },
          fork { expectFile(dut.io.qk_s_stream, dut.clockDomain, s"$dir/CONDENSED_ROPE_QK_BYPASS_S.bin", I8, 1, signed = false, s"ROPE_QK ViT S layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"ROPE_QK ViT layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  def runQK_GEMM(): Unit = {
    simConfig.compile(new QK_GEMM).doSimUntilVoid { dut =>
      init_i_stream(dut.io.qk_q_stream); init_i_stream(dut.io.qk_s_stream)
      init_i_stream(dut.io.kq_cache_i_stream); init_i_stream(dut.io.ks_cache_i_stream)
      init_o_stream(dut.io.kq_cache_o_stream); init_o_stream(dut.io.ks_cache_o_stream); init_o_stream(dut.io.r_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = vitLayerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_QK_GEMM) },
          fork { feedFile(dut.io.qk_q_stream, dut.clockDomain, s"$dir/CONDENSED_ROPE_QK_BYPASS_Q.bin", I8, 8, "QK_GEMM ViT QK_Q") },
          fork { feedFile(dut.io.qk_s_stream, dut.clockDomain, s"$dir/CONDENSED_ROPE_QK_BYPASS_S.bin", I8, 1, "QK_GEMM ViT QK_S") },
          fork { expectFile(dut.io.r_stream, dut.clockDomain, s"$dir/CONDENSED_QK_GEMM_R.bin", I64, 8, signed = true, s"QK_GEMM ViT R layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"QK_GEMM ViT layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  def runSOFTMAX(): Unit = {
    simConfig.compile(new SOFTMAX).doSimUntilVoid { dut =>
      init_i_stream(dut.io.r_stream); init_o_stream(dut.io.rq_stream); init_o_stream(dut.io.rs_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = vitLayerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_SOFTMAX) },
          fork { feedFile(dut.io.r_stream, dut.clockDomain, s"$dir/CONDENSED_QK_GEMM_R.bin", I64, 8, "SOFTMAX ViT R") },
          fork { expectFile(dut.io.rq_stream, dut.clockDomain, s"$dir/CONDENSED_SOFTMAX_R_Q.bin", I8, 8, signed = true,  s"SOFTMAX ViT RQ layer $l") },
          fork { expectFile(dut.io.rs_stream, dut.clockDomain, s"$dir/CONDENSED_SOFTMAX_R_S.bin", I8, 1, signed = false, s"SOFTMAX ViT RS layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"SOFTMAX ViT layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  def runRV_GEMM(): Unit = {
    simConfig.compile(new RV_GEMM).doSimUntilVoid { dut =>
      init_i_stream(dut.io.rq_stream); init_i_stream(dut.io.rs_stream); init_i_stream(dut.io.v_stream)
      init_i_stream(dut.io.vq_cache_i_stream)
      init_i_stream(dut.io.vs_cache_i_stream)
      init_o_stream(dut.io.vq_cache_o_stream)
      init_o_stream(dut.io.vs_cache_o_stream)
      init_o_stream(dut.io.aq_stream); init_o_stream(dut.io.as_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = vitLayerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_RV_GEMM) },
          fork { feedFile(dut.io.rq_stream, dut.clockDomain, s"$dir/CONDENSED_SOFTMAX_R_Q.bin", I8, 8, "RV_GEMM ViT RQ") },
          fork { feedFile(dut.io.rs_stream, dut.clockDomain, s"$dir/CONDENSED_SOFTMAX_R_S.bin", I8, 1, "RV_GEMM ViT RS") },
          fork { feedFile(dut.io.v_stream,  dut.clockDomain, s"$dir/CONDENSED_DEMUX_V.bin",     I64, 8, "RV_GEMM ViT V") },
          fork { expectFile(dut.io.aq_stream, dut.clockDomain, s"$dir/CONDENSED_RV_GEMM_A_Q.bin", I8, 8, signed = true,  s"RV_GEMM ViT A_Q layer $l") },
          fork { expectFile(dut.io.as_stream, dut.clockDomain, s"$dir/CONDENSED_RV_GEMM_A_S.bin", I8, 1, signed = false, s"RV_GEMM ViT A_S layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"RV_GEMM ViT layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  def runSILU_GELU(): Unit = {
    simConfig.compile(new SILU_GELU).doSimUntilVoid { dut =>
      init_i_stream(dut.io.mlp_i_stream); init_o_stream(dut.io.q_stream); init_o_stream(dut.io.s_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = vitLayerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_SILU_GELU) },
          fork { feedFile(dut.io.mlp_i_stream, dut.clockDomain, s"$dir/CONDENSED_DEMUX_FC1.bin", I64, 8, "SILU_GELU ViT FC1") },
          fork { expectFile(dut.io.q_stream, dut.clockDomain, s"$dir/CONDENSED_GELU_XM_Q.bin", I8, 8, signed = true,  s"SILU_GELU ViT XM_Q layer $l") },
          fork { expectFile(dut.io.s_stream, dut.clockDomain, s"$dir/CONDENSED_GELU_XM_S.bin", I8, 1, signed = false, s"SILU_GELU ViT XM_S layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"SILU_GELU ViT layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  def runRMS_LAYERNORM(): Unit = {
    simConfig.compile(new RMS_LAYERNORM).doSimUntilVoid { dut =>
      init_i_stream(dut.io.x_stream); init_i_stream(dut.io.lnw_stream); init_i_stream(dut.io.lnb_stream)
      init_o_stream(dut.io.xlnq_stream); init_o_stream(dut.io.xlns_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = vitLayerDir(l)
        val (lnw, lnb) = vitNormParams(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_RMS_LAYERNORM) },
          fork { feedFile(dut.io.x_stream, dut.clockDomain, s"$dir/CONDENSED_RESIDUAL_O.bin", I64, 8, "RMS_LAYERNORM ViT X") },
          fork { feedValues(dut.io.lnw_stream, dut.clockDomain, lnw, 8, "RMS_LAYERNORM ViT LNW") },
          fork { feedValues(dut.io.lnb_stream, dut.clockDomain, lnb, 8, "RMS_LAYERNORM ViT LNB") },
          fork { expectFile(dut.io.xlnq_stream, dut.clockDomain, s"$dir/CONDENSED_XLN_Q.bin", I8, 8, signed = true,  s"RMS_LAYERNORM ViT Q layer $l") },
          fork { expectFile(dut.io.xlns_stream, dut.clockDomain, s"$dir/CONDENSED_XLN_S.bin", I8, 1, signed = false, s"RMS_LAYERNORM ViT S layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"RMS_LAYERNORM ViT layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  def runRESIDUAL(): Unit = {
    simConfig.compile(new RESIDUAL).doSimUntilVoid { dut =>
      init_i_stream(dut.io.x_stream)
      init_i_stream(dut.io.res_i_stream)
      init_o_stream(dut.io.res_o_stream)
      init_o_stream(dut.io.y_stream)
      startDut(dut.io.signals, dut.clockDomain)

      selectedLayers.foreach { l =>
        val dir = vitLayerDir(l)
        val bin = vitBinaryDir(l)
        val mhaX = vitTokenWindow(s"$bin/MHA_LN_X.bin")
        val mhaORes = vitTokenWindow(s"$bin/MHA_O_RES.bin")
        val mlpX = vitTokenWindow(s"$bin/MLP_LN_X.bin")
        val mlpXfc2Res = vitTokenWindow(s"$bin/MLP_XFC2_RES.bin")
        require(mhaORes.sameElements(mlpX), s"ViT layer $l MHA_O_RES and MLP_LN_X differ; pass1 state reference is ambiguous")

        val xPayload = mhaX ++ vitDeltaOrder(mhaX) ++ mlpX ++ vitDeltaOrder(mlpX)
        val yExpected = vitDeltaOrder(mhaORes) ++ vitDeltaOrder(mlpXfc2Res)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, l, l + 1, paramOp = 0, ipBit = ctrl_cfg.IP_RESIDUAL) },
          fork { feedValues(dut.io.x_stream, dut.clockDomain, xPayload, CP, s"RESIDUAL ViT X replay layer $l") },
          fork { feedFile(dut.io.res_i_stream, dut.clockDomain, s"$dir/CONDENSED_DEMUX_OFC2.bin", I64, CP, s"RESIDUAL ViT delta layer $l") },
          fork { expectFile(dut.io.res_o_stream, dut.clockDomain, s"$dir/CONDENSED_RESIDUAL_O.bin", I64, CP, signed = true, s"RESIDUAL ViT pre-ln layer $l") },
          fork { expectValues(dut.io.y_stream, dut.clockDomain, yExpected, CP, signed = true, s"RESIDUAL ViT writeback layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"RESIDUAL ViT layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  def run(args: Array[String]): Unit = {
    val targets = if (args.isEmpty) Seq("M_AXI") else args.toSeq.map(_.toUpperCase)
    targets.foreach {
      case "M_AXI" => runM_AXI()
      case "WEIGHT_AXI" => runWEIGHT_AXI()
      case "STATE_AXI" => runSTATE_AXI()
      case "RESIDUAL" => runRESIDUAL()
      case "MUX" => runMUX()
      case "PERMUTE" => runPERMUTE()
      case "DEMUX" => runDEMUX()
      case "ROPE_QK" => runROPE_QK()
      case "QK_GEMM" => runQK_GEMM()
      case "SOFTMAX" => runSOFTMAX()
      case "RV_GEMM" => runRV_GEMM()
      case "SILU_GELU" => runSILU_GELU()
      case "RMS_LAYERNORM" => runRMS_LAYERNORM()
      case "STATE_RESIDUAL" =>
        runSTATE_AXI()
        runRESIDUAL()
      case "REST_SINGLE" =>
        runMUX()
        runPERMUTE()
        runDEMUX()
        runROPE_QK()
        runQK_GEMM()
        runSOFTMAX()
        runRV_GEMM()
        runSILU_GELU()
        runRMS_LAYERNORM()
      case other => throw new IllegalArgumentException(s"Unknown ViT payload target: $other")
    }
  }
}

object simulate_vit_payload extends App { VitPayloadSim.run(args) }
object simulate_vit_m_axi extends App { VitPayloadSim.run(Array("M_AXI")) }
object simulate_vit_weight_axi extends App { VitPayloadSim.run(Array("WEIGHT_AXI")) }
object simulate_vit_state_axi extends App { VitPayloadSim.run(Array("STATE_AXI")) }
object simulate_vit_residual extends App { VitPayloadSim.run(Array("RESIDUAL")) }
object simulate_vit_state_residual extends App { VitPayloadSim.run(Array("STATE_RESIDUAL")) }
object simulate_vit_mux extends App { VitPayloadSim.run(Array("MUX")) }
object simulate_vit_permute extends App { VitPayloadSim.run(Array("PERMUTE")) }
object simulate_vit_demux extends App { VitPayloadSim.run(Array("DEMUX")) }
object simulate_vit_rope_qk extends App { VitPayloadSim.run(Array("ROPE_QK")) }
object simulate_vit_qk_gemm extends App { VitPayloadSim.run(Array("QK_GEMM")) }
object simulate_vit_softmax extends App { VitPayloadSim.run(Array("SOFTMAX")) }
object simulate_vit_rv_gemm extends App { VitPayloadSim.run(Array("RV_GEMM")) }
object simulate_vit_silu_gelu extends App { VitPayloadSim.run(Array("SILU_GELU")) }
object simulate_vit_rms_layernorm extends App { VitPayloadSim.run(Array("RMS_LAYERNORM")) }
object simulate_vit_rest_single extends App { VitPayloadSim.run(Array("REST_SINGLE")) }
