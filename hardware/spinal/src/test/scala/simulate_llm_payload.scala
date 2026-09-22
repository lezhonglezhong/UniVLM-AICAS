import spinal.core._
import spinal.core.sim._
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinal.lib.bus.amba4.axis.Axi4Stream.Axi4Stream
import utils._

import java.io.File
import scala.language.postfixOps

// @formatter:off

object LlmPayloadSim {
  // LLM 真实 payload 单模块仿真。输入/输出直接采用 HLS step1 生成的 condensed 文件；
  // memory mover 额外使用 Weights/packed 下的 compact 权重和 norm 参数文件。
  private val LLAMA_L = 32
  private val LLAMA_T = 8
  private val LLAMA_C = 960
  private val LLAMA_CM = 2560
  private val LLAMA_H = 15
  private val LLAMA_HC = 64
  private val LLAMA_HCT = LLAMA_HC / 8
  private val LLAMA_CT = LLAMA_C / 8
  private val LLAMA_CMT = LLAMA_CM / 8
  private val LLAMA_S = 1024
  private val LLAMA_ST = LLAMA_S / 8
  private val LLAMA_VOCAB = 49280
  private val POS = sys.env.getOrElse("VLM_LLM_POS", "96").toInt
  private val TOP_POS = sys.env.get("VLM_LLM_TOP_POS").map(_.toInt).getOrElse(POS + LLAMA_T - 1)
  require(TOP_POS >= POS && TOP_POS < POS + LLAMA_T,
    s"VLM_LLM_TOP_POS=$TOP_POS must stay within the current 8-token tile starting at POS=$POS")
  private val CHUNK = TOP_POS / LLAMA_T
  require(POS / LLAMA_T == CHUNK,
    s"VLM_LLM_POS=$POS and VLM_LLM_TOP_POS=$TOP_POS must refer to the same decode tile")
  private val CHUNK_POS = TOP_POS & (LLAMA_T - 1)
  private val VALID_S = math.min(TOP_POS + 1, LLAMA_S)
  private val VALID_ST = (VALID_S + 7) / 8
  private val VALID_S_PADDED = VALID_ST * 8

  private val NUM_X = LLAMA_T * LLAMA_C
  private val NUM_X2 = 2 * NUM_X
  private val NUM_QK = 2 * LLAMA_T * LLAMA_C
  private val NUM_V = LLAMA_T * LLAMA_C
  private val NUM_UG = 2 * LLAMA_T * LLAMA_CM
  private val NUM_OD = 2 * LLAMA_T * LLAMA_C
  private val NUM_R = LLAMA_H * LLAMA_T * VALID_S_PADDED
  private val NUM_RS = LLAMA_H * LLAMA_T * VALID_ST
  private val NUM_AQ = LLAMA_H * LLAMA_T * LLAMA_HC
  private val NUM_AS = LLAMA_H * LLAMA_T * LLAMA_HCT
  private val NUM_VQ_CUR = LLAMA_H * LLAMA_HC * LLAMA_T
  private val NUM_VS_CUR = LLAMA_H * LLAMA_HC * (LLAMA_T / 8)
  private val NUM_KQ_CUR = LLAMA_H * LLAMA_T * LLAMA_HC
  private val NUM_KS_CUR = LLAMA_H * LLAMA_T * LLAMA_HCT
  private val NUM_KQ_CACHE = LLAMA_H * LLAMA_S * LLAMA_HC
  private val NUM_KS_CACHE = LLAMA_H * LLAMA_S * LLAMA_HCT
  private val NUM_VQ_CACHE = LLAMA_H * LLAMA_HC * LLAMA_S
  private val NUM_VS_CACHE = LLAMA_H * LLAMA_HC * LLAMA_ST
  private val NUM_KQ_CACHE_REPLAY = LLAMA_H * VALID_S * LLAMA_HC
  private val NUM_KS_CACHE_REPLAY = LLAMA_H * VALID_S * LLAMA_HCT
  private val NUM_VQ_CACHE_REPLAY = LLAMA_H * LLAMA_HC * VALID_S_PADDED
  private val NUM_VS_CACHE_REPLAY = LLAMA_H * LLAMA_HC * VALID_ST
  private val DW_CACHE_PACK = 256
  private val DW_CACHE_ENTRY = 128
  private val ENTRIES_PER_WORD = DW_CACHE_PACK / DW_CACHE_ENTRY
  private val DW_AQ = 8
  private val DW_AS = 4
  private val CACHE_PACK_BYTES = DW_CACHE_PACK / 8
  private val K_CACHE_PACKS = LLAMA_L * LLAMA_H * LLAMA_S * LLAMA_HCT
  private val V_CACHE_OFFSET = K_CACHE_PACKS

  private val packedPath = sys.env.getOrElse("VLM_PACKED_WEIGHT_PATH", ctrl_cfg.packed_path)
  private val timeoutCycles = sys.env.getOrElse("VLM_LLM_PAYLOAD_TIMEOUT", "200000000").toInt

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
    if (sys.env.get("VLM_LLM_PAYLOAD_WAVE").contains("1")) base.withFstWave else base
  }

  sealed trait ElemKind { def bytes: Int }
  case object I8 extends ElemKind { val bytes = 1 }
  case object I32 extends ElemKind { val bytes = 4 }
  case object I64 extends ElemKind { val bytes = 8 }

  private def layerDir(l: Int): String = s"${ctrl_cfg.condense_prefix}$l"
  private def binaryDir(l: Int): String = s"${ctrl_cfg.binaries_prefix}$l"
  private def packed(name: String): String = s"$packedPath/$name"

  private def selectedLayers: Seq[Int] =
    sys.env.getOrElse("VLM_LLM_TEST_LAYERS", "0").split("[,\\s]+").filter(_.nonEmpty).map(_.toInt).toSeq

  private def fileElemCount(path: String, kind: ElemKind): Int = {
    val f = new File(path)
    require(f.isFile, s"Missing payload file: $path")
    require(f.length() % kind.bytes == 0, s"File size is not aligned: $path")
    (f.length() / kind.bytes).toInt
  }

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

  private def readTokenWindow(path: String, kind: ElemKind, tokenStart: Int, tokenCount: Int, channels: Int): Array[Long] = {
    // CLS_X 来自 binaries 全窗口文件；这里按 HLS testbench 的 token 窗口裁剪后再喂给 final RMSNorm。
    val all = readValues(path, kind)
    require(all.length % channels == 0, s"$path element count ${all.length} is not divisible by channels=$channels")
    val start = tokenStart * channels
    val end = start + tokenCount * channels
    require(end <= all.length, s"$path token window [$tokenStart, ${tokenStart + tokenCount}) exceeds ${all.length / channels} tokens")
    all.slice(start, end)
  }

  private def readHeadTokenWindow(path: String, kind: ElemKind, tokenStart: Int, tokenCount: Int, heads: Int, channels: Int): Array[Long] = {
    // K/K-scale 二进制参考文件按 H -> T_LOAD -> C 保存；这里裁剪当前 decode tile，
    // 输出顺序保持 H -> T -> C，匹配 QK_GEMM/KV_CACHE 的当前 K 写回流。
    val all = readValues(path, kind)
    require(all.length % (heads * channels) == 0, s"$path element count ${all.length} is not divisible by heads*channels=${heads * channels}")
    val tLoad = all.length / (heads * channels)
    require(tokenStart + tokenCount <= tLoad, s"$path token window [$tokenStart, ${tokenStart + tokenCount}) exceeds $tLoad tokens")
    val out = Array.ofDim[Long](heads * tokenCount * channels)
    for (h <- 0 until heads; t <- 0 until tokenCount; c <- 0 until channels) {
      out(h * tokenCount * channels + t * channels + c) =
        all(h * tLoad * channels + (tokenStart + t) * channels + c)
    }
    out
  }

  private def readHeadTokenWindowPadded(
                                         path: String,
                                         kind: ElemKind,
                                         tokenStart: Int,
                                         tokenCount: Int,
                                         heads: Int,
                                         tLoad: Int,
                                         channelsOut: Int
                                       ): Array[Long] = {
    // SOFTMAX 的 R_Q/R_S 软件文件实际只保存到当前可见 cache 长度 S_LOAD。
    // 旧路径可把短文件补到 S=1024；LLM-01 之后也可只取前 VALID_S/VALID_ST。
    val all = readValues(path, kind)
    require(all.length % (heads * tLoad) == 0, s"$path element count ${all.length} is not divisible by heads*tLoad=${heads * tLoad}")
    val channelsLoad = all.length / (heads * tLoad)
    require(tokenStart + tokenCount <= tLoad, s"$path token window [$tokenStart, ${tokenStart + tokenCount}) exceeds $tLoad tokens")
    val out = Array.ofDim[Long](heads * tokenCount * channelsOut)
    val copyChannels = math.min(channelsLoad, channelsOut)
    for (h <- 0 until heads; t <- 0 until tokenCount; c <- 0 until copyChannels) {
      out(h * tokenCount * channelsOut + t * channelsOut + c) =
        all(h * tLoad * channelsLoad + (tokenStart + t) * channelsLoad + c)
    }
    out
  }

  private def sliceSequence(values: Array[Long], outer: Int, keepSeq: Int, inner: Int, info: String): Array[Long] = {
    // 部分历史 cache condensed 文件按旧 S_LOAD=2000 保存；当前硬件只消费 LLAMA_S=1024。
    // 这里按 outer -> seq -> inner 裁剪前 keepSeq 个位置，保证 Spinal payload 和 HLS 常量一致。
    require(values.length % (outer * inner) == 0, s"$info length ${values.length} is not divisible by outer*inner=${outer * inner}")
    val seqLoad = values.length / (outer * inner)
    require(seqLoad >= keepSeq, s"$info seq length $seqLoad is smaller than required $keepSeq")
    val out = Array.ofDim[Long](outer * keepSeq * inner)
    for (o <- 0 until outer; s <- 0 until keepSeq; i <- 0 until inner) {
      out(o * keepSeq * inner + s * inner + i) = values(o * seqLoad * inner + s * inner + i)
    }
    out
  }

  private def readKqCache(path: String): Array[Long] =
    sliceSequence(readValues(path, I8), LLAMA_H, LLAMA_S, LLAMA_HC, path)

  private def readKsCache(path: String): Array[Long] =
    sliceSequence(readValues(path, I8), LLAMA_H, LLAMA_S, LLAMA_HCT, path)

  private def readKqCacheReplay(path: String): Array[Long] =
    sliceSequence(readValues(path, I8), LLAMA_H, VALID_S, LLAMA_HC, path)

  private def readKsCacheReplay(path: String): Array[Long] =
    sliceSequence(readValues(path, I8), LLAMA_H, VALID_S, LLAMA_HCT, path)

  private def readVqCache(path: String): Array[Long] = {
    // 历史软件 dump 可能保存 S_LOAD=2000 的完整序列；硬件 KV cache 固定 S=1024。
    // V_Q 的顺序是 (H*HC) -> S，因此按 channel 维度裁剪到硬件窗口。
    sliceSequence(readValues(path, I8), LLAMA_H * LLAMA_HC, LLAMA_S, 1, path)
  }

  private def readVsCache(path: String): Array[Long] = {
    // V_S 的顺序是 (H*HC) -> ST，和 V_Q 使用同一个硬件 S 窗口。
    sliceSequence(readValues(path, I8), LLAMA_H * LLAMA_HC, LLAMA_ST, 1, path)
  }

  private def readVqCacheReplay(path: String): Array[Long] =
    sliceSequence(readValues(path, I8), LLAMA_H * LLAMA_HC, VALID_S_PADDED, 1, path)

  private def readVsCacheReplay(path: String): Array[Long] =
    sliceSequence(readValues(path, I8), LLAMA_H * LLAMA_HC, VALID_ST, 1, path)

  private def signExtend(v: BigInt, bits: Int): Long = {
    val sign = BigInt(1) << (bits - 1)
    val mod = BigInt(1) << bits
    val masked = v & (mod - 1)
    (if ((masked & sign) != 0) masked - mod else masked).toLong
  }

  private def unsignedValue(v: Long, bits: Int): Long =
    (BigInt(v) & ((BigInt(1) << bits) - 1)).toLong

  private def normalizedExpected(v: Long, laneBits: Int, signed: Boolean): Long =
    if (signed) signExtend(BigInt(v), laneBits) else unsignedValue(v, laneBits)

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
          if (mismatch <= 10) {
            println(s"$info mismatch at ${index + lane}: got=$got expected=$exp laneBits=$laneBits")
          }
        }
      }
      index += lanes
    }
    stream.ready #= false
    require(mismatch == 0, s"$info has $mismatch mismatches")
    println(s"output $info matched (${expected.length} elements, $lanes lanes)")
  }

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
    // 用于 STATE_AXI layer 闭环验证：只有确认 x_stream 已进入 delta-order 阶段后，
    // y_stream 才开始喂入，避免测试提前提供 y 而掩盖 RTL 中 x/y 互等问题。
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
          if (mismatch <= 10) {
            println(s"$info mismatch at ${index + lane}: got=$got expected=$exp laneBits=$laneBits")
          }
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

  private def expectFile(
                          stream: Axi4Stream,
                          clockDomain: ClockDomain,
                          path: String,
                          kind: ElemKind,
                          lanes: Int,
                          signed: Boolean,
                          info: String
                        ): Unit = expectValues(stream, clockDomain, readValues(path, kind), lanes, signed, info)

  private def drain(
                     stream: Axi4Stream,
                     clockDomain: ClockDomain,
                     elements: Int,
                     lanes: Int,
                     info: String
                   ): Unit = {
    require(elements % lanes == 0, s"$info length $elements is not divisible by lanes=$lanes")
    var index = 0
    stream.ready #= true
    while (index < elements) {
      clockDomain.waitSamplingWhere(stream.valid.toBoolean && stream.ready.toBoolean)
      index += lanes
    }
    stream.ready #= false
    println(s"output $info drained ($elements elements, $lanes lanes)")
  }

  private def monitorStream(
                             name: String,
                             stream: Axi4Stream,
                             clockDomain: ClockDomain,
                             enabled: Boolean
                           ): Unit = {
    // 调试长 payload 仿真时统计 AXIS 握手，确认阻塞点在输入、输出还是控制链。
    if (!enabled) return
    fork {
      var cycles = 0L
      var handshakes = 0L
      var validOnly = 0L
      var readyOnly = 0L
      while (true) {
        clockDomain.waitSampling()
        val valid = stream.valid.toBoolean
        val ready = stream.ready.toBoolean
        if (valid && ready) handshakes += 1
        else if (valid) validOnly += 1
        else if (ready) readyOnly += 1
        cycles += 1
        if (cycles % 1000000L == 0L) {
          println(s"$name monitor cycles=$cycles hs=$handshakes validOnly=$validOnly readyOnly=$readyOnly")
        }
      }
    }
  }

  private def initAndLaunch(
                             signals: DaisyChain[ManagerSignals],
                             clockDomain: ClockDomain,
                             ipBit: Int,
                             lBegin: Int,
                             lClose: Int,
                             paramOp: Int = 0,
                             stateOp: Int = 0,
                             pos: Int = TOP_POS
                           ): Unit = {
    signals.I.MODE #= 0
    signals.I.L_BEGIN #= lBegin
    signals.I.L_CLOSE #= lClose
    signals.I.PARAM_OP #= paramOp
    signals.I.STATE_OP #= stateOp
    signals.I.POS #= pos
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
    // payload 测试需要等待本次启动真正进入 busy；write-only mover 若只检查当前 idle，
    // 可能在 ap_start 生效前提前返回，随后过早读取模拟内存。
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

  private def packQScale(q: Array[Long], qBase: Int, qLanes: Int, s: Long): BigInt = {
    // KV cache memory pack 低位保存 8 个 int8 q lane，随后保存 4-bit scale。
    var pack = BigInt(0)
    for (lane <- 0 until qLanes) {
      pack |= (BigInt(q(qBase + lane)) & ((BigInt(1) << DW_AQ) - 1)) << (DW_AQ * lane)
    }
    pack | ((BigInt(s) & ((BigInt(1) << DW_AS) - 1)) << (DW_AQ * qLanes))
  }

  private def unpackSignedLane(pack: BigInt, lane: Int): Long =
    signExtend((pack >> (DW_AQ * lane)) & ((BigInt(1) << DW_AQ) - 1), DW_AQ)

  private def unpackScale(pack: BigInt, qLanes: Int): Long =
    ((pack >> (DW_AQ * qLanes)) & ((BigInt(1) << DW_AS) - 1)).toLong

  private def writeKCacheEntry(mem: AxiMemorySim, base: Long, entryIdx: Int, entry: BigInt): Unit = {
    // K/V 逻辑 entry 都是 q/s 128-bit，DDR M_AXI word 是 256-bit，因此两个 entry 合并一个 word。
    val wordIdx = entryIdx / ENTRIES_PER_WORD
    val shift = (entryIdx % ENTRIES_PER_WORD) * DW_CACHE_ENTRY
    val addr = base + wordIdx * CACHE_PACK_BYTES
    val oldWord = mem.memory.readBigInt(addr, CACHE_PACK_BYTES)
    val slotMask = ((BigInt(1) << DW_CACHE_ENTRY) - 1) << shift
    val newWord = (oldWord & ~slotMask) | ((entry & ((BigInt(1) << DW_CACHE_ENTRY) - 1)) << shift)
    mem.memory.writeBigInt(addr, newWord, CACHE_PACK_BYTES)
  }

  private def readKCacheEntry(mem: AxiMemorySim, base: Long, entryIdx: Int): BigInt = {
    val wordIdx = entryIdx / ENTRIES_PER_WORD
    val shift = (entryIdx % ENTRIES_PER_WORD) * DW_CACHE_ENTRY
    (mem.memory.readBigInt(base + wordIdx * CACHE_PACK_BYTES, CACHE_PACK_BYTES) >> shift) &
      ((BigInt(1) << DW_CACHE_ENTRY) - 1)
  }

  private def writeKCacheLayer(mem: AxiMemorySim, base: Long, l: Int, kq: Array[Long], ks: Array[Long]): Unit = {
    require(kq.length == NUM_KQ_CACHE, s"KQ cache length ${kq.length} != $NUM_KQ_CACHE")
    require(ks.length == NUM_KS_CACHE, s"KS cache length ${ks.length} != $NUM_KS_CACHE")
    for (h <- 0 until LLAMA_H; s <- 0 until LLAMA_S; hct <- 0 until LLAMA_HCT) {
      val qBase = h * LLAMA_S * LLAMA_HC + s * LLAMA_HC + hct * 8
      val sIdx = h * LLAMA_S * LLAMA_HCT + s * LLAMA_HCT + hct
      val pack = packQScale(kq, qBase, 8, ks(sIdx))
      val packIdx = l * LLAMA_H * LLAMA_S * LLAMA_HCT + h * LLAMA_S * LLAMA_HCT + s * LLAMA_HCT + hct
      writeKCacheEntry(mem, base, packIdx, pack)
    }
  }

  private def writeVCacheLayer(mem: AxiMemorySim, base: Long, l: Int, vq: Array[Long], vs: Array[Long]): Unit = {
    require(vq.length == NUM_VQ_CACHE, s"VQ cache length ${vq.length} != $NUM_VQ_CACHE")
    require(vs.length == NUM_VS_CACHE, s"VS cache length ${vs.length} != $NUM_VS_CACHE")
    for (h <- 0 until LLAMA_H; c <- 0 until LLAMA_HC; st <- 0 until LLAMA_ST) {
      val qBase = h * LLAMA_HC * LLAMA_S + c * LLAMA_S + st * 8
      val sIdx = h * LLAMA_HC * LLAMA_ST + c * LLAMA_ST + st
      val pack = packQScale(vq, qBase, 8, vs(sIdx))
      val packIdx = V_CACHE_OFFSET + l * LLAMA_H * LLAMA_HC * LLAMA_ST + h * LLAMA_HC * LLAMA_ST + c * LLAMA_ST + st
      writeKCacheEntry(mem, base, packIdx, pack)
    }
  }

  private def expectKCacheWriteback(mem: AxiMemorySim, base: Long, l: Int, kq: Array[Long], ks: Array[Long], info: String): Unit = {
    var mismatch = 0
    for (h <- 0 until LLAMA_H; t <- 0 to CHUNK_POS; hct <- 0 until LLAMA_HCT) {
      val s = POS + t
      val packIdx = l * LLAMA_H * LLAMA_S * LLAMA_HCT + h * LLAMA_S * LLAMA_HCT + s * LLAMA_HCT + hct
      val pack = readKCacheEntry(mem, base, packIdx)
      for (cp <- 0 until 8) {
        val idx = h * LLAMA_T * LLAMA_HC + t * LLAMA_HC + hct * 8 + cp
        val got = unpackSignedLane(pack, cp)
        val exp = normalizedExpected(kq(idx), DW_AQ, signed = true)
        if (got != exp) {
          mismatch += 1
          if (mismatch <= 10) println(s"$info KQ mismatch at t=$t idx=$idx: got=$got expected=$exp")
        }
      }
      val sIdx = h * LLAMA_T * LLAMA_HCT + t * LLAMA_HCT + hct
      val gotS = unpackScale(pack, 8)
      val expS = normalizedExpected(ks(sIdx), DW_AS, signed = false)
      if (gotS != expS) {
        mismatch += 1
        if (mismatch <= 10) println(s"$info KS mismatch at t=$t idx=$sIdx: got=$gotS expected=$expS")
      }
    }
    require(mismatch == 0, s"$info has $mismatch K-cache writeback mismatches")
  }

  private def expectVCacheWriteback(mem: AxiMemorySim, base: Long, l: Int, vq: Array[Long], vs: Array[Long], info: String): Unit = {
    require(vq.length == NUM_VQ_CUR, s"VQ current length ${vq.length} != $NUM_VQ_CUR")
    require(vs.length == NUM_VS_CUR, s"VS current length ${vs.length} != $NUM_VS_CUR")
    var mismatch = 0
    for (h <- 0 until LLAMA_H; c <- 0 until LLAMA_HC) {
      val st = CHUNK
      val packIdx = V_CACHE_OFFSET + l * LLAMA_H * LLAMA_HC * LLAMA_ST + h * LLAMA_HC * LLAMA_ST + c * LLAMA_ST + st
      val pack = readKCacheEntry(mem, base, packIdx)
      for (sp <- 0 until 8) {
        val idx = h * LLAMA_HC * LLAMA_T + c * LLAMA_T + sp
        val got = unpackSignedLane(pack, sp)
        val exp = normalizedExpected(vq(idx), DW_AQ, signed = true)
        if (got != exp) {
          mismatch += 1
          if (mismatch <= 10) println(s"$info VQ mismatch at c=$c sp=$sp idx=$idx: got=$got expected=$exp")
        }
      }
      val sIdx = h * LLAMA_HC * (LLAMA_T / 8) + c * (LLAMA_T / 8)
      val gotS = unpackScale(pack, 8)
      val expS = normalizedExpected(vs(sIdx), DW_AS, signed = false)
      if (gotS != expS) {
        mismatch += 1
        if (mismatch <= 10) println(s"$info VS mismatch at c=$c idx=$sIdx: got=$gotS expected=$expS")
      }
    }
    require(mismatch == 0, s"$info has $mismatch V-cache writeback mismatches")
  }

  private def llmDeltaOrder(values: Array[Long]): Array[Long] = {
    // RESIDUAL 的 LLM add/writeback 顺序为 CT -> token -> CP；memory/state 文件本身是 token-major。
    require(values.length == NUM_X, s"delta-order source length ${values.length} != $NUM_X")
    val out = Array.ofDim[Long](NUM_X)
    var idx = 0
    for (ct <- 0 until LLAMA_CT; t <- 0 until LLAMA_T; cp <- 0 until 8) {
      out(idx) = values(t * LLAMA_C + ct * 8 + cp)
      idx += 1
    }
    out
  }

  private def maybeMonitorAxiWrites(name: String, dut: STATE_AXI): Unit = {
    // 调试 STATE_AXI/CLS 窄写时使用；默认关闭，避免大量 AXI 日志淹没数值比对输出。
    if (!sys.env.get("VLM_LLM_PAYLOAD_DEBUG").contains("1")) return
    fork {
      var awSeen = 0
      var wSeen = 0
      while (awSeen < 16 || wSeen < 16) {
        dut.clockDomain.waitSampling()
        if (dut.io.gmem1.aw.valid.toBoolean && dut.io.gmem1.aw.ready.toBoolean && awSeen < 16) {
          println(
            s"$name AW[$awSeen] addr=0x${dut.io.gmem1.aw.payload.addr.toBigInt.toString(16)} " +
              s"len=${dut.io.gmem1.aw.payload.len.toBigInt} size=${dut.io.gmem1.aw.payload.size.toBigInt} " +
              s"burst=${dut.io.gmem1.aw.payload.burst.toBigInt}"
          )
          awSeen += 1
        }
        if (dut.io.gmem1.w.valid.toBoolean && dut.io.gmem1.w.ready.toBoolean && wSeen < 16) {
          println(
            s"$name W[$wSeen] data=0x${dut.io.gmem1.w.payload.data.toBigInt.toString(16)} " +
              s"strb=0x${dut.io.gmem1.w.payload.strb.toBigInt.toString(16)} last=${dut.io.gmem1.w.payload.last.toBoolean}"
          )
          wSeen += 1
        }
        if (dut.io.gmem1.b.valid.toBoolean && dut.io.gmem1.b.ready.toBoolean) {
          println(s"$name B")
        }
      }
    }
  }

  private def splitPackedToMem(memLo: AxiMemorySim, memHi: AxiMemorySim, baseLo: Long, baseHi: Long, path: String): Unit = {
    val bytes = read_int8_file(path, fileElemCount(path, I8))
    require(bytes.length % 2 == 0, s"Packed weight file must have even length: $path")
    val half = bytes.length / 2
    writeBytes(memLo, baseLo, bytes.slice(0, half))
    writeBytes(memHi, baseHi, bytes.slice(half, bytes.length))
  }

  private def runM_AXI(): Unit = {
    simConfig.compile(new M_AXI).doSimUntilVoid { dut =>
      val axi = AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig())
      axi.reset()
      init_o_stream(dut.io.bias_stream)
      init_o_stream(dut.io.lnw_stream)
      init_o_stream(dut.io.lnb_stream)
      startDut(dut.io.signals, dut.clockDomain)

      val llmBase = 0x1000L
      val clsBase = 0x100000L
      writeBytes(axi, llmBase, read_int8_file(packed("all_llm_lnw_i32.bin"), fileElemCount(packed("all_llm_lnw_i32.bin"), I8)))
      writeBytes(axi, clsBase, read_int8_file(packed("cls_lnw_i32.bin"), fileElemCount(packed("cls_lnw_i32.bin"), I8)))

      val allLlm = readValues(packed("all_llm_lnw_i32.bin"), I32)
      val cls = readValues(packed("cls_lnw_i32.bin"), I32)

      selectedLayers.foreach { l =>
        dut.io.signals.I.MEMORY_LLM_LNW #= llmBase
        dut.io.signals.I.MEMORY_CLS_LNW #= clsBase
        val ref = allLlm.slice(l * 2 * LLAMA_C, (l + 1) * 2 * LLAMA_C)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_M_AXI, l, l + 1, paramOp = 1) },
          fork { expectValues(dut.io.lnw_stream, dut.clockDomain, ref, 8, signed = true, s"M_AXI LLM lnw layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"M_AXI layer $l") }
        ).foreach(_.join())
      }

      dut.io.signals.I.MEMORY_LLM_LNW #= llmBase
      dut.io.signals.I.MEMORY_CLS_LNW #= clsBase
      Array(
        fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_M_AXI, LLAMA_L, LLAMA_L + 1, paramOp = 1) },
        fork { expectValues(dut.io.lnw_stream, dut.clockDomain, cls, 8, signed = true, "M_AXI CLS lnw") },
        fork { waitIdle(dut.clockDomain, dut.io.idle, "M_AXI CLS") }
      ).foreach(_.join())
      simSuccess()
    }
  }

  private def runWEIGHT_AXI(): Unit = {
    simConfig.compile(new WEIGHT_AXI).doSimUntilVoid { dut =>
      val axiLo = AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig())
      val axiHi = AxiMemorySim(dut.io.gmem2, dut.clockDomain, AxiMemorySimConfig())
      axiLo.reset()
      axiHi.reset()
      init_o_stream(dut.io.wq_stream)
      init_o_stream(dut.io.ws1_stream)
      init_o_stream(dut.io.ws2_stream)
      startDut(dut.io.signals, dut.clockDomain)

      val decBase = 0x1000L
      val clsBase = 0x20000000L
      splitPackedToMem(axiLo, axiHi, decBase, decBase, packed("all_decoder_w.bin"))
      splitPackedToMem(axiLo, axiHi, clsBase, clsBase, packed("all_cls_w.bin"))

      selectedLayers.foreach { l =>
        dut.io.signals.I.MEMORY_DECODER_W_LO #= decBase
        dut.io.signals.I.MEMORY_DECODER_W_HI #= decBase
        dut.io.signals.I.MEMORY_CLS_W_LO #= clsBase
        dut.io.signals.I.MEMORY_CLS_W_HI #= clsBase
        val dir = layerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_WEIGHT_AXI, l, l + 1) },
          fork { expectFile(dut.io.wq_stream,  dut.clockDomain, s"$dir/CONDENSED_GEMM_W_Q.bin",  I8, 64, signed = true,  s"WEIGHT_AXI WQ layer $l") },
          fork { expectFile(dut.io.ws1_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_W_S1.bin", I8, 8,  signed = false, s"WEIGHT_AXI WS1 layer $l") },
          fork { expectFile(dut.io.ws2_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_W_S2.bin", I8, 8,  signed = false, s"WEIGHT_AXI WS2 layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"WEIGHT_AXI layer $l") }
        ).foreach(_.join())
      }

      dut.io.signals.I.MEMORY_DECODER_W_LO #= decBase
      dut.io.signals.I.MEMORY_DECODER_W_HI #= decBase
      dut.io.signals.I.MEMORY_CLS_W_LO #= clsBase
      dut.io.signals.I.MEMORY_CLS_W_HI #= clsBase
      val clsDir = layerDir(LLAMA_L)
      Array(
        fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_WEIGHT_AXI, LLAMA_L, LLAMA_L + 1) },
        fork { expectFile(dut.io.wq_stream,  dut.clockDomain, s"$clsDir/CONDENSED_GEMM_W_Q.bin",  I8, 64, signed = true,  "WEIGHT_AXI CLS WQ") },
        fork { expectFile(dut.io.ws1_stream, dut.clockDomain, s"$clsDir/CONDENSED_GEMM_W_S1.bin", I8, 8,  signed = false, "WEIGHT_AXI CLS WS1") },
        fork { expectFile(dut.io.ws2_stream, dut.clockDomain, s"$clsDir/CONDENSED_GEMM_W_S2.bin", I8, 8,  signed = false, "WEIGHT_AXI CLS WS2") },
        fork { waitIdle(dut.clockDomain, dut.io.idle, "WEIGHT_AXI CLS") }
      ).foreach(_.join())
      simSuccess()
    }
  }

  private def runSTATE_AXI(): Unit = {
    simConfig.compile(new STATE_AXI).doSimUntilVoid { dut =>
      val axi = AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig())
      axi.reset()
      init_o_stream(dut.io.x_stream)
      init_i_stream(dut.io.y_stream)
      init_i_stream(dut.io.cls_stream)
      startDut(dut.io.signals, dut.clockDomain)

      val stateBase = 0x1000L
      val clsBase = 0x100000L
      selectedLayers.foreach { l =>
        val refX = readValues(s"${layerDir(l)}/CONDENSED_RESIDUAL_X.bin", I64)
        writeI32Values(axi, stateBase, refX)
        dut.io.signals.I.MEMORY_DECODER_STATE #= stateBase
        dut.io.signals.I.MEMORY_CLS_Y #= clsBase
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_STATE_AXI, l, l + 1, stateOp = 0) },
          fork { expectValues(dut.io.x_stream, dut.clockDomain, refX, 8, signed = true, s"STATE_AXI token replay layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"STATE_AXI token layer $l") }
        ).foreach(_.join())

        val bin = binaryDir(l)
        val mhaX = readTokenWindow(s"$bin/MHA_X.bin", I64, POS, LLAMA_T, LLAMA_C)
        val mhaORes = readTokenWindow(s"$bin/MHA_O_RES.bin", I64, POS, LLAMA_T, LLAMA_C)
        val mlpXdRes = readTokenWindow(s"$bin/MLP_XD_RES.bin", I64, POS, LLAMA_T, LLAMA_C)
        // AXI_STATE_LAYER 是顶层 decoder layer 使用的闭合调度：STATE_AXI 依次发
        // pass0 token-major、pass0 delta-order、pass1 token-major、pass1 delta-order，
        // 并在两段 delta-order 中消费 RESIDUAL 的 y_stream 写回 memory。
        val layerXExpected = mhaX ++ llmDeltaOrder(mhaX) ++ mhaORes ++ llmDeltaOrder(mhaORes)
        val layerYPayload = llmDeltaOrder(mhaORes) ++ llmDeltaOrder(mlpXdRes)
        @volatile var sawDeltaReplay = false
        writeI32Values(axi, stateBase, mhaX)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_STATE_AXI, l, l + 1, stateOp = 4) },
          fork { expectValuesWithMarker(dut.io.x_stream, dut.clockDomain, layerXExpected, 8, signed = true, s"STATE_AXI layer x replay $l", markerElements = NUM_X + 8, mark = () => sawDeltaReplay = true) },
          fork {
            var cycles = 0
            while (!sawDeltaReplay && cycles < timeoutCycles) {
              dut.clockDomain.waitSampling()
              cycles += 1
            }
            require(sawDeltaReplay, s"STATE_AXI layer $l did not emit delta-order x before y feed")
            feedValues(dut.io.y_stream, dut.clockDomain, layerYPayload, 8, "STATE_AXI layer y writeback input")
          },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"STATE_AXI layer op $l") }
        ).foreach(_.join())
        val layerGot = readI32Values(axi, stateBase, NUM_X)
        require(layerGot.sameElements(mlpXdRes), s"STATE_AXI layer memory mismatch for layer $l")
      }

      val clsRef = readValues(s"${layerDir(LLAMA_L)}/CONDENSED_DEMUX_CLS_INDEX.bin", I64)
      dut.io.signals.I.MEMORY_DECODER_STATE #= stateBase
      dut.io.signals.I.MEMORY_CLS_Y #= clsBase
      // CLS 写回是 128-bit AXI 上的 int32 窄写；先预分配模拟内存页，避免页缺失掩盖真实写回行为。
      writeI32Values(axi, clsBase, Array.fill[Long](LLAMA_T)(0))
      maybeMonitorAxiWrites("STATE_AXI CLS", dut)
      Array(
        fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_STATE_AXI, LLAMA_L, LLAMA_L + 1, stateOp = 3) },
        fork { feedValues(dut.io.cls_stream, dut.clockDomain, clsRef, 8, "STATE_AXI CLS index input") },
        fork { waitIdle(dut.clockDomain, dut.io.idle, "STATE_AXI CLS write") }
      ).foreach(_.join())
      val got = readI32Values(axi, clsBase, LLAMA_T)
      require(got.sameElements(clsRef), s"STATE_AXI CLS memory mismatch: got=${got.mkString(",")} expected=${clsRef.mkString(",")}")
      simSuccess()
    }
  }

  private def runKV_CACHE(): Unit = {
    simConfig.compile(new KV_CACHE).doSimUntilVoid { dut =>
      val axi = AxiMemorySim(dut.io.gmem1, dut.clockDomain, AxiMemorySimConfig())
      axi.reset()
      init_o_stream(dut.io.kq_cache_i_stream); init_i_stream(dut.io.kq_cache_o_stream)
      init_o_stream(dut.io.ks_cache_i_stream); init_i_stream(dut.io.ks_cache_o_stream)
      init_o_stream(dut.io.vq_cache_i_stream); init_i_stream(dut.io.vq_cache_o_stream)
      init_o_stream(dut.io.vs_cache_i_stream); init_i_stream(dut.io.vs_cache_o_stream)
      startDut(dut.io.signals, dut.clockDomain)

      val cacheBase = 0x1000L
      selectedLayers.foreach { l =>
        val dir = layerDir(l)
        val bin = binaryDir(l)
        val kqCache = readKqCache(s"$dir/CONDENSED_KQ_CACHE.bin")
        val ksCache = readKsCache(s"$dir/CONDENSED_KS_CACHE.bin")
        val vqCache = readVqCache(s"$dir/CONDENSED_RV_GEMM_V_Q_CACHE.bin")
        val vsCache = readVsCache(s"$dir/CONDENSED_RV_GEMM_V_S_CACHE.bin")
        val kqReplay = readKqCacheReplay(s"$dir/CONDENSED_KQ_CACHE.bin")
        val ksReplay = readKsCacheReplay(s"$dir/CONDENSED_KS_CACHE.bin")
        val vqReplay = readVqCacheReplay(s"$dir/CONDENSED_RV_GEMM_V_Q_CACHE.bin")
        val vsReplay = readVsCacheReplay(s"$dir/CONDENSED_RV_GEMM_V_S_CACHE.bin")
        val kqCur = readHeadTokenWindow(s"$bin/MHA_K_Q.bin", I8, POS, LLAMA_T, LLAMA_H, LLAMA_HC)
        val ksCur = readHeadTokenWindow(s"$bin/MHA_K_S.bin", I8, POS, LLAMA_T, LLAMA_H, LLAMA_HCT)
        val vqCur = readValues(s"$dir/CONDENSED_RV_GEMM_V_Q.bin", I8)
        val vsCur = readValues(s"$dir/CONDENSED_RV_GEMM_V_S.bin", I8)

        writeKCacheLayer(axi, cacheBase, l, kqCache, ksCache)
        writeVCacheLayer(axi, cacheBase, l, vqCache, vsCache)
        dut.io.signals.I.MEMORY_K_CACHE #= cacheBase
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_KV_CACHE, l, l + 1) },
          fork { expectValues(dut.io.kq_cache_i_stream, dut.clockDomain, kqReplay, 8, signed = true,  s"KV_CACHE KQ replay layer $l") },
          fork { expectValues(dut.io.ks_cache_i_stream, dut.clockDomain, ksReplay, 1, signed = false, s"KV_CACHE KS replay layer $l") },
          fork { expectValues(dut.io.vq_cache_i_stream, dut.clockDomain, vqReplay, 8, signed = true,  s"KV_CACHE VQ replay layer $l") },
          fork { expectValues(dut.io.vs_cache_i_stream, dut.clockDomain, vsReplay, 1, signed = false, s"KV_CACHE VS replay layer $l") },
          fork { feedValues(dut.io.kq_cache_o_stream, dut.clockDomain, kqCur, 8, "KV_CACHE KQ current input") },
          fork { feedValues(dut.io.ks_cache_o_stream, dut.clockDomain, ksCur, 1, "KV_CACHE KS current input") },
          fork { feedValues(dut.io.vq_cache_o_stream, dut.clockDomain, vqCur, 8, "KV_CACHE VQ writeback input") },
          fork { feedValues(dut.io.vs_cache_o_stream, dut.clockDomain, vsCur, 1, "KV_CACHE VS writeback input") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"KV_CACHE layer $l") }
        ).foreach(_.join())
        expectKCacheWriteback(axi, cacheBase, l, kqCur, ksCur, s"KV_CACHE layer $l")
        expectVCacheWriteback(axi, cacheBase, l, vqCur, vsCur, s"KV_CACHE layer $l")
      }
      simSuccess()
    }
  }

  private def runMUX(): Unit = {
    simConfig.compile(new MUX).doSimUntilVoid { dut =>
      init_i_stream(dut.io.xlnq_stream); init_i_stream(dut.io.xlns_stream)
      init_i_stream(dut.io.aq_stream);   init_i_stream(dut.io.as_stream)
      init_i_stream(dut.io.xmq_stream);  init_i_stream(dut.io.xms_stream)
      init_o_stream(dut.io.q_stream);    init_o_stream(dut.io.s_stream)
      startDut(dut.io.signals, dut.clockDomain)
      val debugMux = sys.env.get("VLM_LLM_PAYLOAD_DEBUG").contains("1")
      monitorStream("MUX xlnq", dut.io.xlnq_stream, dut.clockDomain, debugMux)
      monitorStream("MUX xlns", dut.io.xlns_stream, dut.clockDomain, debugMux)
      monitorStream("MUX aq", dut.io.aq_stream, dut.clockDomain, debugMux)
      monitorStream("MUX as", dut.io.as_stream, dut.clockDomain, debugMux)
      monitorStream("MUX xmq", dut.io.xmq_stream, dut.clockDomain, debugMux)
      monitorStream("MUX xms", dut.io.xms_stream, dut.clockDomain, debugMux)
      monitorStream("MUX q", dut.io.q_stream, dut.clockDomain, debugMux)
      monitorStream("MUX s", dut.io.s_stream, dut.clockDomain, debugMux)

      selectedLayers.foreach { l =>
        val dir = layerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_MUX, l, l + 1) },
          fork { feedFile(dut.io.xlnq_stream, dut.clockDomain, s"$dir/CONDENSED_XLN_Q.bin", I8, 8, "MUX XLN_Q") },
          fork { feedFile(dut.io.xlns_stream, dut.clockDomain, s"$dir/CONDENSED_XLN_S.bin", I8, 1, "MUX XLN_S") },
          fork { feedFile(dut.io.aq_stream,   dut.clockDomain, s"$dir/CONDENSED_RV_GEMM_A_Q.bin", I8, 8, "MUX A_Q") },
          fork { feedFile(dut.io.as_stream,   dut.clockDomain, s"$dir/CONDENSED_RV_GEMM_A_S.bin", I8, 1, "MUX A_S") },
          fork { feedFile(dut.io.xmq_stream,  dut.clockDomain, s"$dir/CONDENSED_SILU_EM_QUANT_XM_Q.bin", I8, 8, "MUX XM_Q") },
          fork { feedFile(dut.io.xms_stream,  dut.clockDomain, s"$dir/CONDENSED_SILU_EM_QUANT_XM_S.bin", I8, 1, "MUX XM_S") },
          fork { expectFile(dut.io.q_stream,  dut.clockDomain, s"$dir/CONDENSED_GEMM_X_Q.bin", I8, 64, signed = true,  s"MUX X_Q layer $l") },
          fork { expectFile(dut.io.s_stream,  dut.clockDomain, s"$dir/CONDENSED_GEMM_X_S.bin", I8, 8,  signed = false, s"MUX X_S layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"MUX layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  private def runPERMUTE(): Unit = {
    simConfig.compile(new PERMUTE).doSimUntilVoid { dut =>
      init_i_stream(dut.io.i_stream); init_i_stream(dut.io.s_stream); init_i_stream(dut.io.w_stream)
      init_i_stream(dut.io.s1_stream); init_i_stream(dut.io.s2_stream); init_o_stream(dut.io.o_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = layerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_PERMUTE, l, l + 1) },
          fork { feedFile(dut.io.i_stream,  dut.clockDomain, s"$dir/CONDENSED_GEMM_X_Q.bin",  I8, 64, "PERMUTE X_Q") },
          fork { feedFile(dut.io.s_stream,  dut.clockDomain, s"$dir/CONDENSED_GEMM_X_S.bin",  I8, 8,  "PERMUTE X_S") },
          fork { feedFile(dut.io.w_stream,  dut.clockDomain, s"$dir/CONDENSED_GEMM_W_Q.bin",  I8, 64, "PERMUTE W_Q") },
          fork { feedFile(dut.io.s1_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_W_S1.bin", I8, 8,  "PERMUTE W_S1") },
          fork { feedFile(dut.io.s2_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_W_S2.bin", I8, 8,  "PERMUTE W_S2") },
          fork { expectFile(dut.io.o_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_Y.bin", I64, 8, signed = true, s"PERMUTE Y layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"PERMUTE layer $l") }
        ).foreach(_.join())
      }

      val clsDir = layerDir(LLAMA_L)
      Array(
        fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_PERMUTE, LLAMA_L, LLAMA_L + 1) },
        fork { feedFile(dut.io.i_stream,  dut.clockDomain, s"$clsDir/CONDENSED_GEMM_X_Q.bin",  I8, 64, "PERMUTE CLS X_Q") },
        fork { feedFile(dut.io.s_stream,  dut.clockDomain, s"$clsDir/CONDENSED_GEMM_X_S.bin",  I8, 8,  "PERMUTE CLS X_S") },
        fork { feedFile(dut.io.w_stream,  dut.clockDomain, s"$clsDir/CONDENSED_GEMM_W_Q.bin",  I8, 64, "PERMUTE CLS W_Q") },
        fork { feedFile(dut.io.s1_stream, dut.clockDomain, s"$clsDir/CONDENSED_GEMM_W_S1.bin", I8, 8,  "PERMUTE CLS W_S1") },
        fork { feedFile(dut.io.s2_stream, dut.clockDomain, s"$clsDir/CONDENSED_GEMM_W_S2.bin", I8, 8,  "PERMUTE CLS W_S2") },
        fork { expectFile(dut.io.o_stream, dut.clockDomain, s"$clsDir/CONDENSED_GEMM_Y.bin", I64, 8, signed = true, "PERMUTE CLS Y") },
        fork { waitIdle(dut.clockDomain, dut.io.idle, "PERMUTE CLS") }
      ).foreach(_.join())
      simSuccess()
    }
  }

  private def runDEMUX(): Unit = {
    simConfig.compile(new DEMUX).doSimUntilVoid { dut =>
      init_i_stream(dut.io.gemm_stream); init_i_stream(dut.io.bias_stream)
      init_o_stream(dut.io.qk_stream); init_o_stream(dut.io.v_stream); init_o_stream(dut.io.mlp1_stream)
      init_o_stream(dut.io.od_fc2_stream); init_o_stream(dut.io.cls_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = layerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_DEMUX, l, l + 1, pos = POS + LLAMA_T - 1) },
          fork { feedFile(dut.io.gemm_stream, dut.clockDomain, s"$dir/CONDENSED_GEMM_Y.bin", I64, 8, "DEMUX GEMM_Y") },
          fork { expectFile(dut.io.qk_stream,     dut.clockDomain, s"$dir/CONDENSED_DEMUX_QK.bin", I64, 8, signed = true, s"DEMUX QK layer $l") },
          fork { expectFile(dut.io.v_stream,      dut.clockDomain, s"$dir/CONDENSED_DEMUX_V.bin",  I64, 8, signed = true, s"DEMUX V layer $l") },
          fork { expectFile(dut.io.mlp1_stream,   dut.clockDomain, s"$dir/CONDENSED_DEMUX_UG.bin", I64, 8, signed = true, s"DEMUX UG layer $l") },
          fork { expectFile(dut.io.od_fc2_stream, dut.clockDomain, s"$dir/CONDENSED_DEMUX_OD.bin", I64, 8, signed = true, s"DEMUX OD layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"DEMUX layer $l") }
        ).foreach(_.join())
      }
      val clsDir = layerDir(LLAMA_L)
      Array(
        fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_DEMUX, LLAMA_L, LLAMA_L + 1, pos = POS + LLAMA_T - 1) },
        fork { feedFile(dut.io.gemm_stream, dut.clockDomain, s"$clsDir/CONDENSED_GEMM_Y.bin", I64, 8, "DEMUX CLS GEMM_Y") },
        fork { expectFile(dut.io.cls_stream, dut.clockDomain, s"$clsDir/CONDENSED_DEMUX_CLS_INDEX.bin", I64, 8, signed = true, "DEMUX CLS index") },
        fork { waitIdle(dut.clockDomain, dut.io.idle, "DEMUX CLS") }
      ).foreach(_.join())
      simSuccess()
    }
  }

  private def runROPE_QK(): Unit = {
    simConfig.compile(new ROPE_QK).doSimUntilVoid { dut =>
      init_i_stream(dut.io.qk_i_stream); init_o_stream(dut.io.qk_q_stream); init_o_stream(dut.io.qk_s_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = layerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_ROPE_QK, l, l + 1) },
          fork { feedFile(dut.io.qk_i_stream, dut.clockDomain, s"$dir/CONDENSED_DEMUX_QK.bin", I64, 8, "ROPE_QK input") },
          fork { expectFile(dut.io.qk_q_stream, dut.clockDomain, s"$dir/CONDENSED_ROPE_QK_QUANT_ROT_Q.bin", I8, 8, signed = true,  s"ROPE_QK Q layer $l") },
          fork { expectFile(dut.io.qk_s_stream, dut.clockDomain, s"$dir/CONDENSED_ROPE_QK_QUANT_ROT_S.bin", I8, 1, signed = false, s"ROPE_QK S layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"ROPE_QK layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  private def runSOFTMAX(): Unit = {
    simConfig.compile(new SOFTMAX).doSimUntilVoid { dut =>
      init_i_stream(dut.io.r_stream); init_o_stream(dut.io.rq_stream); init_o_stream(dut.io.rs_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = layerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_SOFTMAX, l, l + 1) },
          fork { feedFile(dut.io.r_stream, dut.clockDomain, s"$dir/CONDENSED_QK_GEMM_R.bin", I64, 8, "SOFTMAX R") },
          fork { expectFile(dut.io.rq_stream, dut.clockDomain, s"$dir/CONDENSED_SOFTMAX_QUANT_R_Q.bin", I8, 8, signed = true,  s"SOFTMAX RQ layer $l") },
          fork { expectFile(dut.io.rs_stream, dut.clockDomain, s"$dir/CONDENSED_SOFTMAX_QUANT_R_S.bin", I8, 1, signed = false, s"SOFTMAX RS layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"SOFTMAX layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  private def runSILU_GELU(): Unit = {
    simConfig.compile(new SILU_GELU).doSimUntilVoid { dut =>
      init_i_stream(dut.io.mlp_i_stream); init_o_stream(dut.io.q_stream); init_o_stream(dut.io.s_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = layerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_SILU_GELU, l, l + 1) },
          fork { feedFile(dut.io.mlp_i_stream, dut.clockDomain, s"$dir/CONDENSED_DEMUX_UG.bin", I64, 8, "SILU_GELU UG") },
          fork { expectFile(dut.io.q_stream, dut.clockDomain, s"$dir/CONDENSED_SILU_EM_QUANT_XM_Q.bin", I8, 8, signed = true,  s"SILU_GELU XM_Q layer $l") },
          fork { expectFile(dut.io.s_stream, dut.clockDomain, s"$dir/CONDENSED_SILU_EM_QUANT_XM_S.bin", I8, 1, signed = false, s"SILU_GELU XM_S layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"SILU_GELU layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  private def runRMS_LAYERNORM(): Unit = {
    simConfig.compile(new RMS_LAYERNORM).doSimUntilVoid { dut =>
      init_i_stream(dut.io.x_stream); init_i_stream(dut.io.lnw_stream); init_i_stream(dut.io.lnb_stream)
      init_o_stream(dut.io.xlnq_stream); init_o_stream(dut.io.xlns_stream)
      startDut(dut.io.signals, dut.clockDomain)
      val allLlm = readValues(packed("all_llm_lnw_i32.bin"), I32)
      val cls = readValues(packed("cls_lnw_i32.bin"), I32)
      selectedLayers.foreach { l =>
        val dir = layerDir(l)
        val lnw = allLlm.slice(l * 2 * LLAMA_C, (l + 1) * 2 * LLAMA_C)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_RMS_LAYERNORM, l, l + 1) },
          fork { feedFile(dut.io.x_stream, dut.clockDomain, s"$dir/CONDENSED_RESIDUAL_O.bin", I64, 8, "RMS_LAYERNORM X") },
          fork { feedValues(dut.io.lnw_stream, dut.clockDomain, lnw, 8, "RMS_LAYERNORM LNW") },
          fork { expectFile(dut.io.xlnq_stream, dut.clockDomain, s"$dir/CONDENSED_XLN_Q.bin", I8, 8, signed = true,  s"RMS_LAYERNORM Q layer $l") },
          fork { expectFile(dut.io.xlns_stream, dut.clockDomain, s"$dir/CONDENSED_XLN_S.bin", I8, 1, signed = false, s"RMS_LAYERNORM S layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"RMS_LAYERNORM layer $l") }
        ).foreach(_.join())
      }
      val clsDir = layerDir(LLAMA_L)
      // CLS/lm_head 单模块 payload 与 decoder+CLS 顶层统一验证当前 POS 窗口。
      val clsX = readTokenWindow(s"${binaryDir(LLAMA_L)}/CLS_X.bin", I64, POS, LLAMA_T, LLAMA_C)
      Array(
        fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_RMS_LAYERNORM, LLAMA_L, LLAMA_L + 1) },
        fork { feedValues(dut.io.x_stream, dut.clockDomain, clsX, 8, "RMS_LAYERNORM CLS X") },
        fork { feedValues(dut.io.lnw_stream, dut.clockDomain, cls, 8, "RMS_LAYERNORM CLS LNW") },
        fork { expectFile(dut.io.xlnq_stream, dut.clockDomain, s"$clsDir/CONDENSED_XLN_Q.bin", I8, 8, signed = true,  "RMS_LAYERNORM CLS Q") },
        fork { expectFile(dut.io.xlns_stream, dut.clockDomain, s"$clsDir/CONDENSED_XLN_S.bin", I8, 1, signed = false, "RMS_LAYERNORM CLS S") },
        fork { waitIdle(dut.clockDomain, dut.io.idle, "RMS_LAYERNORM CLS") }
      ).foreach(_.join())
      simSuccess()
    }
  }

  private def runRESIDUAL(): Unit = {
    simConfig.compile(new RESIDUAL).doSimUntilVoid { dut =>
      init_i_stream(dut.io.x_stream); init_i_stream(dut.io.res_i_stream)
      init_o_stream(dut.io.res_o_stream); init_o_stream(dut.io.y_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = layerDir(l)
        val bin = binaryDir(l)
        val mhaX = readTokenWindow(s"$bin/MHA_X.bin", I64, POS, LLAMA_T, LLAMA_C)
        val mlpX = readTokenWindow(s"$bin/MLP_X.bin", I64, POS, LLAMA_T, LLAMA_C)
        val mhaORes = readTokenWindow(s"$bin/MHA_O_RES.bin", I64, POS, LLAMA_T, LLAMA_C)
        val mlpXdRes = readTokenWindow(s"$bin/MLP_XD_RES.bin", I64, POS, LLAMA_T, LLAMA_C)
        // RESIDUAL 一次 layer 调用需要四段 x：pass0 token-major、pass0 delta-order、
        // pass1 token-major、pass1 delta-order；writeback 输出保持 delta-order。
        val xPayload = mhaX ++ llmDeltaOrder(mhaX) ++ mlpX ++ llmDeltaOrder(mlpX)
        val yExpected = llmDeltaOrder(mhaORes) ++ llmDeltaOrder(mlpXdRes)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_RESIDUAL, l, l + 1) },
          fork { feedValues(dut.io.x_stream, dut.clockDomain, xPayload, 8, "RESIDUAL X replay") },
          fork { feedFile(dut.io.res_i_stream, dut.clockDomain, s"$dir/CONDENSED_DEMUX_OD.bin", I64, 8, "RESIDUAL delta") },
          fork { expectFile(dut.io.res_o_stream, dut.clockDomain, s"$dir/CONDENSED_RESIDUAL_O.bin", I64, 8, signed = true, s"RESIDUAL pre-ln layer $l") },
          fork { expectValues(dut.io.y_stream, dut.clockDomain, yExpected, 8, signed = true, s"RESIDUAL writeback layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"RESIDUAL layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  private def runQK_GEMM(): Unit = {
    simConfig.compile(new QK_GEMM).doSimUntilVoid { dut =>
      init_i_stream(dut.io.qk_q_stream); init_i_stream(dut.io.qk_s_stream)
      init_i_stream(dut.io.kq_cache_i_stream); init_i_stream(dut.io.ks_cache_i_stream)
      init_o_stream(dut.io.kq_cache_o_stream); init_o_stream(dut.io.ks_cache_o_stream); init_o_stream(dut.io.r_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = layerDir(l)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_QK_GEMM, l, l + 1) },
          fork { feedFile(dut.io.qk_q_stream,       dut.clockDomain, s"$dir/CONDENSED_ROPE_QK_QUANT_ROT_Q.bin", I8, 8, "QK_GEMM QK_Q") },
          fork { feedFile(dut.io.qk_s_stream,       dut.clockDomain, s"$dir/CONDENSED_ROPE_QK_QUANT_ROT_S.bin", I8, 1, "QK_GEMM QK_S") },
          fork { feedValues(dut.io.kq_cache_i_stream, dut.clockDomain, readKqCacheReplay(s"$dir/CONDENSED_KQ_CACHE.bin"), 8, "QK_GEMM KQ_CACHE") },
          fork { feedValues(dut.io.ks_cache_i_stream, dut.clockDomain, readKsCacheReplay(s"$dir/CONDENSED_KS_CACHE.bin"), 1, "QK_GEMM KS_CACHE") },
          fork { drain(dut.io.kq_cache_o_stream, dut.clockDomain, NUM_KQ_CUR, 8, "QK_GEMM KQ current") },
          fork { drain(dut.io.ks_cache_o_stream, dut.clockDomain, NUM_KS_CUR, 1, "QK_GEMM KS current") },
          fork { expectFile(dut.io.r_stream, dut.clockDomain, s"$dir/CONDENSED_QK_GEMM_R.bin", I64, 8, signed = true, s"QK_GEMM R layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"QK_GEMM layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  private def runRV_GEMM(): Unit = {
    simConfig.compile(new RV_GEMM).doSimUntilVoid { dut =>
      init_i_stream(dut.io.rq_stream); init_i_stream(dut.io.rs_stream); init_i_stream(dut.io.v_stream)
      init_i_stream(dut.io.vq_cache_i_stream)
      init_i_stream(dut.io.vs_cache_i_stream)
      init_o_stream(dut.io.vq_cache_o_stream)
      init_o_stream(dut.io.vs_cache_o_stream)
      init_o_stream(dut.io.aq_stream); init_o_stream(dut.io.as_stream)
      startDut(dut.io.signals, dut.clockDomain)
      selectedLayers.foreach { l =>
        val dir = layerDir(l)
        val bin = binaryDir(l)
        val vTLoad = fileElemCount(s"$bin/MHA_V_SPLIT_HEADS.bin", I64) / (LLAMA_H * LLAMA_HC)
        val rq = readHeadTokenWindowPadded(s"$bin/MHA_R_Q.bin", I8, POS, LLAMA_T, LLAMA_H, vTLoad, VALID_S_PADDED)
        val rs = readHeadTokenWindowPadded(s"$bin/MHA_R_S.bin", I8, POS, LLAMA_T, LLAMA_H, vTLoad, VALID_ST)
        Array(
          fork { initAndLaunch(dut.io.signals, dut.clockDomain, ctrl_cfg.IP_RV_GEMM, l, l + 1) },
          fork { feedValues(dut.io.rq_stream, dut.clockDomain, rq, 8, "RV_GEMM RQ") },
          fork { feedValues(dut.io.rs_stream, dut.clockDomain, rs, 1, "RV_GEMM RS") },
          fork { feedFile(dut.io.v_stream,  dut.clockDomain, s"$dir/CONDENSED_DEMUX_V.bin", I64, 8, "RV_GEMM V") },
          fork { feedValues(dut.io.vq_cache_i_stream, dut.clockDomain, readVqCacheReplay(s"$dir/CONDENSED_RV_GEMM_V_Q_CACHE.bin"), 8, "RV_GEMM VQ_CACHE") },
          fork { feedValues(dut.io.vs_cache_i_stream, dut.clockDomain, readVsCacheReplay(s"$dir/CONDENSED_RV_GEMM_V_S_CACHE.bin"), 1, "RV_GEMM VS_CACHE") },
          fork { expectFile(dut.io.vq_cache_o_stream, dut.clockDomain, s"$dir/CONDENSED_RV_GEMM_V_Q.bin", I8, 8, signed = true, s"RV_GEMM VQ layer $l") },
          fork { expectFile(dut.io.vs_cache_o_stream, dut.clockDomain, s"$dir/CONDENSED_RV_GEMM_V_S.bin", I8, 1, signed = false, s"RV_GEMM VS layer $l") },
          fork { expectFile(dut.io.aq_stream, dut.clockDomain, s"$dir/CONDENSED_RV_GEMM_A_Q.bin", I8, 8, signed = true,  s"RV_GEMM AQ layer $l") },
          fork { expectFile(dut.io.as_stream, dut.clockDomain, s"$dir/CONDENSED_RV_GEMM_A_S.bin", I8, 1, signed = false, s"RV_GEMM AS layer $l") },
          fork { waitIdle(dut.clockDomain, dut.io.idle, s"RV_GEMM layer $l") }
        ).foreach(_.join())
      }
      simSuccess()
    }
  }

  private val runners: Seq[(String, () => Unit)] = Seq(
    "M_AXI" -> (() => runM_AXI()),
    "WEIGHT_AXI" -> (() => runWEIGHT_AXI()),
    "STATE_AXI" -> (() => runSTATE_AXI()),
    "KV_CACHE" -> (() => runKV_CACHE()),
    "MUX" -> (() => runMUX()),
    "PERMUTE" -> (() => runPERMUTE()),
    "DEMUX" -> (() => runDEMUX()),
    "ROPE_QK" -> (() => runROPE_QK()),
    "QK_GEMM" -> (() => runQK_GEMM()),
    "SOFTMAX" -> (() => runSOFTMAX()),
    "RV_GEMM" -> (() => runRV_GEMM()),
    "SILU_GELU" -> (() => runSILU_GELU()),
    "RESIDUAL" -> (() => runRESIDUAL()),
    "RMS_LAYERNORM" -> (() => runRMS_LAYERNORM())
  )

  def run(args: Array[String]): Unit = {
    val selected =
      if (args.isEmpty) runners
      else {
        val wanted = args.map(_.toUpperCase).toSet
        runners.filter { case (name, _) => wanted.contains(name) }
      }
    require(selected.nonEmpty, s"No LLM payload simulation matched args: ${args.mkString(",")}")
    selected.foreach { case (name, runner) =>
      println(s"Running LLM payload simulation for $name, layers=${selectedLayers.mkString(",")}, pos=$POS, top_pos=$TOP_POS, valid_s=$VALID_S")
      runner()
    }
  }
}

object simulate_llm_payload extends App { LlmPayloadSim.run(args) }
object simulate_llm_m_axi extends App { LlmPayloadSim.run(Array("M_AXI")) }
object simulate_llm_weight_axi extends App { LlmPayloadSim.run(Array("WEIGHT_AXI")) }
object simulate_llm_state_axi extends App { LlmPayloadSim.run(Array("STATE_AXI")) }
object simulate_llm_kv_cache extends App { LlmPayloadSim.run(Array("KV_CACHE")) }
object simulate_llm_mux extends App { LlmPayloadSim.run(Array("MUX")) }
object simulate_llm_permute extends App { LlmPayloadSim.run(Array("PERMUTE")) }
object simulate_llm_demux extends App { LlmPayloadSim.run(Array("DEMUX")) }
object simulate_llm_rope_qk extends App { LlmPayloadSim.run(Array("ROPE_QK")) }
object simulate_llm_qk_gemm extends App { LlmPayloadSim.run(Array("QK_GEMM")) }
object simulate_llm_softmax extends App { LlmPayloadSim.run(Array("SOFTMAX")) }
object simulate_llm_rv_gemm extends App { LlmPayloadSim.run(Array("RV_GEMM")) }
object simulate_llm_silu_gelu extends App { LlmPayloadSim.run(Array("SILU_GELU")) }
object simulate_llm_residual extends App { LlmPayloadSim.run(Array("RESIDUAL")) }
object simulate_llm_rms_layernorm extends App { LlmPayloadSim.run(Array("RMS_LAYERNORM")) }

// @formatter:on
