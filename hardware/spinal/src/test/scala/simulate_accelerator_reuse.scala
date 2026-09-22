import spinal.core._
import spinal.core.sim._
import spinal.sim.SimThread
import spinal.lib.bus.amba4.axi.sim.{AxiMemorySim, AxiMemorySimConfig}
import spinal.lib.bus.amba4.axis.Axi4Stream.Axi4Stream
import utils._

import java.io.File
import scala.language.postfixOps

// @formatter:off

object AcceleratorReusePayloadSim {
  // 完整 ACCELERATOR 顶层 LLM 数值仿真。
  // decoder layer 使用 AXI_STATE_LAYER，让 STATE_AXI 在一次 layer 调用内交替完成
  // token replay 与 delta replay/writeback；CLS/lm_head 另起 WRITE_CLS 消费 DEMUX argmax。
  private val LLAMA_L = 32
  private val LLAMA_T = 8
  private val LLAMA_C = 960
  private val LLAMA_H = 15
  private val LLAMA_KVH = 5
  private val LLAMA_GQA = LLAMA_H / LLAMA_KVH
  private val LLAMA_HC = 64
  private val LLAMA_HCT = LLAMA_HC / 8
  private val LLAMA_CT = LLAMA_C / 8
  private val LLAMA_S = 1024
  private val LLAMA_ST = LLAMA_S / 8
  private val POS = sys.env.getOrElse("VLM_LLM_POS", "96").toInt
  private val TOP_POS = sys.env.get("VLM_LLM_TOP_POS").map(_.toInt).getOrElse(POS + LLAMA_T - 1)
  private val CHUNK = POS / LLAMA_T

  private val NUM_X = LLAMA_T * LLAMA_C
  private val NUM_KQ_CACHE = LLAMA_KVH * LLAMA_S * LLAMA_HC
  private val NUM_KS_CACHE = LLAMA_KVH * LLAMA_S * LLAMA_HCT
  private val NUM_VQ_CACHE = LLAMA_KVH * LLAMA_HC * LLAMA_S
  private val NUM_VS_CACHE = LLAMA_KVH * LLAMA_HC * LLAMA_ST
  private val DW_CACHE_PACK = 256
  private val DW_CACHE_ENTRY = 128
  private val ENTRIES_PER_WORD = DW_CACHE_PACK / DW_CACHE_ENTRY
  private val DW_AQ = 8
  private val DW_AS = 4
  private val CACHE_PACK_BYTES = DW_CACHE_PACK / 8
  private val K_CACHE_PACKS = LLAMA_L * LLAMA_KVH * LLAMA_S * LLAMA_HCT
  private val V_CACHE_OFFSET = K_CACHE_PACKS
  private val STATE_LAYER_OP = 4
  private val VIT_L = 12
  private val VIT_T = 1024
  private val VIT_C = 768
  private val VIT_NUM_X = VIT_T * VIT_C

  private val PARAM_LLM_LNW = 0x1000L
  private val PARAM_CLS_LNW = 0x100000L
  private val PARAM_VIT_BIAS = 0x200000L
  private val PARAM_VIT_LNW = 0x300000L
  private val PARAM_VIT_LNB = 0x400000L
  private val WEIGHT_DEC = 0x1000L
  private val WEIGHT_CLS = 0x20000000L
  private val WEIGHT_VIT = 0x1000L
  private val STATE_DEC = 0x1000L
  private val STATE_CLS_Y = 0x100000L
  private val STATE_VIT = 0x1000L
  private val VIT_A = 0x1000L
  private val VIT_XM = 0x1000L
  private val KV_BASE = 0x1000L

  private val timeoutCycles = sys.env.getOrElse("VLM_ACCEL_TIMEOUT", "200000000").toInt
  private val heartbeatCycles = sys.env.getOrElse("VLM_ACCEL_HEARTBEAT", "0").toInt
  private val debugInternal = sys.env.get("VLM_ACCEL_DEBUG").contains("1")
  private val checkStreams = sys.env.get("VLM_ACCEL_CHECK_STREAMS").contains("1")
  private val clsWriteDelay = sys.env.getOrElse("VLM_ACCEL_CLS_WRITE_DELAY", "50000").toInt
  private val clsStateOverride = sys.env.get("VLM_ACCEL_CLS_STATE_PATH")
  private val clsExpectedOverride = sys.env.get("VLM_ACCEL_CLS_EXPECTED_PATH")
  private val clsDebugDirOverride = sys.env.get("VLM_ACCEL_CLS_DEBUG_DIR")

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
    if (sys.env.get("VLM_ACCEL_WAVE").contains("1")) base.withFstWave else base
  }

  sealed trait ElemKind { def bytes: Int }
  case object I8 extends ElemKind { val bytes = 1 }
  case object I32 extends ElemKind { val bytes = 4 }
  case object I64 extends ElemKind { val bytes = 8 }

  private def layerDir(l: Int): String = s"${ctrl_cfg.condense_prefix}$l"
  private def binaryDir(l: Int): String = s"${ctrl_cfg.binaries_prefix}$l"
  private def vitLayerDir(l: Int): String = s"${ctrl_cfg.vit_condense_prefix}$l"
  private def vitBinaryDir(l: Int): String = s"${ctrl_cfg.weights_root}/ViT/binaries/vision_$l"
  private def packed(name: String): String = s"${ctrl_cfg.packed_path}/$name"

  private def mask(bit: Int): BigInt = BigInt(1) << bit
  private val decoderMask =
    mask(ctrl_cfg.IP_M_AXI) | mask(ctrl_cfg.IP_WEIGHT_AXI) | mask(ctrl_cfg.IP_KV_CACHE) |
    mask(ctrl_cfg.IP_STATE_AXI) | mask(ctrl_cfg.IP_MUX) | mask(ctrl_cfg.IP_PERMUTE) |
    mask(ctrl_cfg.IP_DEMUX) | mask(ctrl_cfg.IP_ROPE_QK) | mask(ctrl_cfg.IP_QK_GEMM) |
    mask(ctrl_cfg.IP_SOFTMAX) | mask(ctrl_cfg.IP_RV_GEMM) | mask(ctrl_cfg.IP_SILU_GELU) |
    mask(ctrl_cfg.IP_RESIDUAL) | mask(ctrl_cfg.IP_RMS_LAYERNORM)
  private val clsMainMask =
    mask(ctrl_cfg.IP_M_AXI) | mask(ctrl_cfg.IP_WEIGHT_AXI) | mask(ctrl_cfg.IP_STATE_AXI) |
    mask(ctrl_cfg.IP_MUX) | mask(ctrl_cfg.IP_PERMUTE) | mask(ctrl_cfg.IP_DEMUX) |
    mask(ctrl_cfg.IP_RESIDUAL) | mask(ctrl_cfg.IP_RMS_LAYERNORM)
  private val vitMainMask =
    mask(ctrl_cfg.IP_M_AXI) | mask(ctrl_cfg.IP_WEIGHT_AXI) | mask(ctrl_cfg.IP_STATE_AXI) |
    mask(ctrl_cfg.IP_MUX) | mask(ctrl_cfg.IP_PERMUTE) | mask(ctrl_cfg.IP_DEMUX) |
    mask(ctrl_cfg.IP_ROPE_QK) | mask(ctrl_cfg.IP_QK_GEMM) | mask(ctrl_cfg.IP_SOFTMAX) |
    mask(ctrl_cfg.IP_RV_GEMM) | mask(ctrl_cfg.IP_SILU_GELU) |
    mask(ctrl_cfg.IP_A_REORDER) | mask(ctrl_cfg.IP_XM_REORDER) |
    mask(ctrl_cfg.IP_RESIDUAL) | mask(ctrl_cfg.IP_RMS_LAYERNORM)

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

  private def unsignedValue(v: Long, bits: Int): Long =
    (BigInt(v) & ((BigInt(1) << bits) - 1)).toLong

  private def normalizedExpected(v: Long, laneBits: Int, signed: Boolean): Long =
    if (signed) signExtend(BigInt(v), laneBits) else unsignedValue(v, laneBits)

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

  private def readTokenWindow(path: String, tokenStart: Int, tokenCount: Int, channels: Int): Array[Long] = {
    // 与 HLS testbench 保持一致：T_LOAD 从实际文件长度推导，避免旧 612 token 假设静默截断。
    val all = readValues(path, I64)
    require(all.length % channels == 0, s"$path element count ${all.length} is not divisible by $channels")
    val tLoad = all.length / channels
    require(tokenStart + tokenCount <= tLoad, s"$path token window [$tokenStart, ${tokenStart + tokenCount}) exceeds $tLoad tokens")
    all.slice(tokenStart * channels, (tokenStart + tokenCount) * channels)
  }

  private def readVitWindow(path: String): Array[Long] = {
    // ViT 激活 binary 当前按 T_LOAD=13312 保存；PL layer 只处理前 VIT_T=1024 token。
    val all = readValues(path, I64)
    require(all.length % VIT_C == 0, s"$path element count ${all.length} is not divisible by VIT_C=$VIT_C")
    require(all.length >= VIT_NUM_X, s"$path has ${all.length} elements, smaller than ViT state $VIT_NUM_X")
    all.slice(0, VIT_NUM_X)
  }

  private def llmDeltaOrder(values: Array[Long]): Array[Long] = {
    // RESIDUAL/DEMUX 的 LLM delta/writeback 顺序为 CT -> token -> CP；
    // state 文件和最终 memory 仍保持 token-major。
    require(values.length == NUM_X, s"delta-order source length ${values.length} != $NUM_X")
    val out = Array.ofDim[Long](NUM_X)
    var idx = 0
    for (ct <- 0 until LLAMA_CT; t <- 0 until LLAMA_T; cp <- 0 until 8) {
      out(idx) = values(t * LLAMA_C + ct * 8 + cp)
      idx += 1
    }
    out
  }

  private def sliceSequence(values: Array[Long], outer: Int, keepSeq: Int, inner: Int, info: String): Array[Long] = {
    // 历史 cache 文件可能按旧 S_LOAD=2000 保存；当前复用 IP 的 KV cache 常量是 LLAMA_S=1024。
    require(values.length % (outer * inner) == 0, s"$info length ${values.length} is not divisible by outer*inner=${outer * inner}")
    val seqLoad = values.length / (outer * inner)
    require(seqLoad >= keepSeq, s"$info seq length $seqLoad is smaller than required $keepSeq")
    val out = Array.ofDim[Long](outer * keepSeq * inner)
    for (o <- 0 until outer; s <- 0 until keepSeq; i <- 0 until inner) {
      out(o * keepSeq * inner + s * inner + i) = values(o * seqLoad * inner + s * inner + i)
    }
    out
  }

  private def readHeadMajorGqaCache(path: String, keepSeq: Int, inner: Int): Array[Long] = {
    val values = readValues(path, I8)
    val expandedStride = LLAMA_H * inner
    if (values.length % expandedStride == 0 && values.length / expandedStride >= keepSeq) {
      val seqLoad = values.length / expandedStride
      val out = Array.ofDim[Long](LLAMA_KVH * keepSeq * inner)
      for (kv <- 0 until LLAMA_KVH; s <- 0 until keepSeq; i <- 0 until inner) {
        val srcHead = kv * LLAMA_GQA
        out(kv * keepSeq * inner + s * inner + i) = values(srcHead * seqLoad * inner + s * inner + i)
      }
      out
    } else {
      sliceSequence(values, LLAMA_KVH, keepSeq, inner, path)
    }
  }

  private def readKqCache(path: String): Array[Long] =
    readHeadMajorGqaCache(path, LLAMA_S, LLAMA_HC)

  private def readKsCache(path: String): Array[Long] =
    readHeadMajorGqaCache(path, LLAMA_S, LLAMA_HCT)

  private def readVqCache(path: String): Array[Long] = {
    // 历史软件 dump 可能保存 S_LOAD=2000 的完整序列；硬件 KV cache 固定 S=1024。
    // V_Q 的顺序是 (H*HC) -> S，因此按 channel 维度裁剪到硬件窗口。
    readChannelMajorGqaCache(path, LLAMA_S)
  }

  private def readVsCache(path: String): Array[Long] = {
    // V_S 的顺序是 (H*HC) -> ST，和 V_Q 使用同一个硬件 S 窗口。
    readChannelMajorGqaCache(path, LLAMA_ST)
  }

  private def readChannelMajorGqaCache(path: String, keepSeq: Int): Array[Long] = {
    val values = readValues(path, I8)
    val expandedChannels = LLAMA_H * LLAMA_HC
    if (values.length % expandedChannels == 0 && values.length / expandedChannels >= keepSeq) {
      val seqLoad = values.length / expandedChannels
      val out = Array.ofDim[Long](LLAMA_KVH * LLAMA_HC * keepSeq)
      for (kv <- 0 until LLAMA_KVH; c <- 0 until LLAMA_HC; s <- 0 until keepSeq) {
        val srcHead = kv * LLAMA_GQA
        val srcChannel = srcHead * LLAMA_HC + c
        out(kv * LLAMA_HC * keepSeq + c * keepSeq + s) = values(srcChannel * seqLoad + s)
      }
      out
    } else {
      sliceSequence(values, LLAMA_KVH * LLAMA_HC, keepSeq, 1, path)
    }
  }

  private def writeBytes(mem: AxiMemorySim, base: Long, bytes: Array[Byte]): Unit =
    for (i <- bytes.indices) mem.memory.write(base + i, bytes(i))

  private def writeI32Values(mem: AxiMemorySim, base: Long, values: Array[Long]): Unit = {
    for (i <- values.indices) mem.memory.writeBigInt(base + i * 4L, BigInt(values(i)) & 0xffffffffL, 4)
  }

  private def readI32Values(mem: AxiMemorySim, base: Long, count: Int): Array[Long] = {
    val out = Array.ofDim[Long](count)
    for (i <- 0 until count) out(i) = signExtend(mem.memory.readBigInt(base + i * 4L, 4), 32)
    out
  }

  private def writeCondensedWeight(memLo: AxiMemorySim, memHi: AxiMemorySim, baseLo: Long, baseHi: Long, layerOffset: Long, path: String): Unit = {
    // WEIGHT_AXI 的 LO/HI 两个 128-bit bundle 使用 Qwen 风格交织拆分。
    val bytes = read_int8_file(path, fileElemCount(path, I8))
    require(bytes.length % 32 == 0, s"Condensed weight size must align to 256-bit cycles: $path")
    val half = bytes.length / 2
    val bytePerHalf = 16
    val cycles = bytes.length / 32
    for (cyc <- 0 until cycles; loHi <- 0 until 2; n <- 0 until bytePerHalf) {
      val b = bytes(cyc * 32 + loHi * bytePerHalf + n)
      val addr = layerOffset + cyc * bytePerHalf + n
      if (loHi == 0) memLo.memory.write(baseLo + addr, b) else memHi.memory.write(baseHi + addr, b)
    }
    require(layerOffset == 0 || half > 0)
  }

  private def decoderWeightHalfBytes: Long =
    fileElemCount(packed("all_decoder_w.bin"), I8).toLong / 2 / LLAMA_L

  private def writeDecoderPackedLayer(weightLo: AxiMemorySim, weightHi: AxiMemorySim, l: Int): Unit = {
    // Native-GQA decoder weights are packed as all_layer_lo_half ++ all_layer_hi_half.
    val bytes = read_int8_file(packed("all_decoder_w.bin"), fileElemCount(packed("all_decoder_w.bin"), I8))
    require(bytes.length % (2 * LLAMA_L) == 0, "all_decoder_w.bin must contain equal LO/HI halves for every LLM layer")
    val totalHalf = bytes.length / 2
    val layerHalf = totalHalf / LLAMA_L
    val loBase = l * layerHalf
    val hiBase = totalHalf + l * layerHalf
    val layerOffset = l.toLong * layerHalf
    for (i <- 0 until layerHalf) {
      weightLo.memory.write(WEIGHT_DEC + layerOffset + i, bytes(loBase + i))
      weightHi.memory.write(WEIGHT_DEC + layerOffset + i, bytes(hiBase + i))
    }
  }

  private def writeNormParams(paramMem: AxiMemorySim): Unit = {
    writeBytes(paramMem, PARAM_LLM_LNW, read_int8_file(packed("all_llm_lnw_i32.bin"), fileElemCount(packed("all_llm_lnw_i32.bin"), I8)))
    writeBytes(paramMem, PARAM_CLS_LNW, read_int8_file(packed("cls_lnw_i32.bin"), fileElemCount(packed("cls_lnw_i32.bin"), I8)))
  }

  private def writeVitParams(paramMem: AxiMemorySim): Unit = {
    writeBytes(paramMem, PARAM_VIT_BIAS, read_int8_file(packed("all_vit_bias_i32.bin"), fileElemCount(packed("all_vit_bias_i32.bin"), I8)))
    writeBytes(paramMem, PARAM_VIT_LNW,  read_int8_file(packed("all_vit_lnw_i32.bin"),  fileElemCount(packed("all_vit_lnw_i32.bin"),  I8)))
    writeBytes(paramMem, PARAM_VIT_LNB,  read_int8_file(packed("all_vit_lnb_i64.bin"),  fileElemCount(packed("all_vit_lnb_i64.bin"),  I8)))
  }

  private def writeVitPackedWeights(weightLo: AxiMemorySim, weightHi: AxiMemorySim): Unit = {
    // all_vit_w.bin 已按 all_layer_lo_half ++ all_layer_hi_half 保存，直接拆到两路 AXI memory。
    val bytes = read_int8_file(packed("all_vit_w.bin"), fileElemCount(packed("all_vit_w.bin"), I8))
    require(bytes.length % 2 == 0, "all_vit_w.bin must have even LO/HI halves")
    val half = bytes.length / 2
    for (i <- 0 until half) {
      weightLo.memory.write(WEIGHT_VIT + i, bytes(i))
      weightHi.memory.write(WEIGHT_VIT + i, bytes(half + i))
    }
  }

  private def packQScale(q: Array[Long], qBase: Int, qLanes: Int, s: Long): BigInt = {
    var pack = BigInt(0)
    for (lane <- 0 until qLanes) pack |= (BigInt(q(qBase + lane)) & ((BigInt(1) << DW_AQ) - 1)) << (DW_AQ * lane)
    pack | ((BigInt(s) & ((BigInt(1) << DW_AS) - 1)) << (DW_AQ * qLanes))
  }

  private def writeKCacheEntry(mem: AxiMemorySim, entryIdx: Int, entry: BigInt): Unit = {
    // K/V cache 都是 128-bit q/s entry，两个 entry 合并写入一个 256-bit DDR word。
    val wordIdx = entryIdx / ENTRIES_PER_WORD
    val shift = (entryIdx % ENTRIES_PER_WORD) * DW_CACHE_ENTRY
    val addr = KV_BASE + wordIdx * CACHE_PACK_BYTES
    val oldWord = mem.memory.readBigInt(addr, CACHE_PACK_BYTES)
    val slotMask = ((BigInt(1) << DW_CACHE_ENTRY) - 1) << shift
    val newWord = (oldWord & ~slotMask) | ((entry & ((BigInt(1) << DW_CACHE_ENTRY) - 1)) << shift)
    mem.memory.writeBigInt(addr, newWord, CACHE_PACK_BYTES)
  }

  private def writeKCacheLayer(mem: AxiMemorySim, l: Int): Unit = {
    val kq = readKqCache(s"${layerDir(l)}/CONDENSED_KQ_CACHE.bin")
    val ks = readKsCache(s"${layerDir(l)}/CONDENSED_KS_CACHE.bin")
    require(kq.length == NUM_KQ_CACHE && ks.length == NUM_KS_CACHE, s"Unexpected K cache length for layer $l")
    for (h <- 0 until LLAMA_KVH; s <- 0 until LLAMA_S; hct <- 0 until LLAMA_HCT) {
      val qBase = h * LLAMA_S * LLAMA_HC + s * LLAMA_HC + hct * 8
      val sIdx = h * LLAMA_S * LLAMA_HCT + s * LLAMA_HCT + hct
      val packIdx = l * LLAMA_KVH * LLAMA_S * LLAMA_HCT + h * LLAMA_S * LLAMA_HCT + s * LLAMA_HCT + hct
      writeKCacheEntry(mem, packIdx, packQScale(kq, qBase, 8, ks(sIdx)))
    }
  }

  private def writeVCacheLayer(mem: AxiMemorySim, l: Int): Unit = {
    val vq = readVqCache(s"${layerDir(l)}/CONDENSED_RV_GEMM_V_Q_CACHE.bin")
    val vs = readVsCache(s"${layerDir(l)}/CONDENSED_RV_GEMM_V_S_CACHE.bin")
    for (h <- 0 until LLAMA_KVH; c <- 0 until LLAMA_HC; st <- 0 until LLAMA_ST) {
      val qBase = h * LLAMA_HC * LLAMA_S + c * LLAMA_S + st * 8
      val sIdx = h * LLAMA_HC * LLAMA_ST + c * LLAMA_ST + st
      val packIdx = V_CACHE_OFFSET + l * LLAMA_KVH * LLAMA_HC * LLAMA_ST + h * LLAMA_HC * LLAMA_ST + c * LLAMA_ST + st
      writeKCacheEntry(mem, packIdx, packQScale(vq, qBase, 8, vs(sIdx)))
    }
  }

  private def loadLayerInputs(weightLo: AxiMemorySim, weightHi: AxiMemorySim, kvMem: AxiMemorySim, l: Int): Unit = {
    writeDecoderPackedLayer(weightLo, weightHi, l)
    writeKCacheLayer(kvMem, l)
    writeVCacheLayer(kvMem, l)
  }

  private def loadClsWeight(weightLo: AxiMemorySim, weightHi: AxiMemorySim): Unit =
    writeCondensedWeight(weightLo, weightHi, WEIGHT_CLS, WEIGHT_CLS, 0, s"${layerDir(LLAMA_L)}/CONDENSED_WEIGHT.bin")

  private def loadInitialState(stateMem: AxiMemorySim, l: Int): Unit =
    writeI32Values(stateMem, STATE_DEC, readTokenWindow(s"${binaryDir(l)}/MHA_X.bin", POS, LLAMA_T, LLAMA_C))

  private def clsInputState(): Array[Long] =
    clsStateOverride match {
      // 调试板端 CLS/lm_head 差异时，可直接喂真实 8-token final decoder state，
      // 避免为了单个 pos 覆盖全局 Weights/LLM/binaries/decoder_32/CLS_X.bin。
      case Some(path) => readValues(path, I32)
      case None       => readTokenWindow(s"${binaryDir(LLAMA_L)}/CLS_X.bin", POS, LLAMA_T, LLAMA_C)
    }

  private def clsExpectedIndex(): Array[Long] =
    clsExpectedOverride match {
      case Some(path) => readValues(path, I32)
      case None       => readValues(s"${layerDir(LLAMA_L)}/CONDENSED_DEMUX_CLS_INDEX.bin", I64)
    }

  private def clsDebugFile(name: String): String =
    clsDebugDirOverride match {
      // 自定义 CLS/lm_head state 调试时，中间参考文件放在独立目录，避免覆盖
      // Weights/LLM/condense/decoder_32 下的 POS=96 常规 step1 产物。
      case Some(dir) => s"$dir/$name"
      case None      => s"${layerDir(LLAMA_L)}/$name"
    }

  private def loadClsInitialState(stateMem: AxiMemorySim): Unit =
    // CLS/lm_head 单独测试也使用当前 decode POS 窗口，和 decoder+CLS 顶层参考保持一致。
    writeI32Values(stateMem, STATE_DEC, clsInputState())

  private def compareState(stateMem: AxiMemorySim, expected: Array[Long], info: String): Unit = {
    val got = readI32Values(stateMem, STATE_DEC, NUM_X)
    var mismatch = 0
    for (i <- got.indices) {
      if (got(i) != expected(i)) {
        mismatch += 1
        if (mismatch <= 10) println(s"$info mismatch at $i: got=${got(i)} expected=${expected(i)}")
      }
    }
    require(mismatch == 0, s"$info has $mismatch mismatches")
    println(s"$info matched ($NUM_X int32 values)")
  }

  private def compareLayerOutput(stateMem: AxiMemorySim, l: Int): Unit =
    compareState(stateMem, readValues(s"${layerDir(l)}/CONDENSED_RESIDUAL_Y.bin", I64), s"ACCELERATOR decoder layer $l state")

  private def compareClsInput(stateMem: AxiMemorySim): Unit =
    compareState(stateMem, clsInputState(), "ACCELERATOR CLS input state")

  private def compareClsOutput(stateMem: AxiMemorySim): Unit = {
    val expected = clsExpectedIndex()
    val got = readI32Values(stateMem, STATE_CLS_Y, LLAMA_T)
    require(got.sameElements(expected), s"ACCELERATOR CLS index mismatch: got=${got.mkString(",")} expected=${expected.mkString(",")}")
    println(s"ACCELERATOR CLS index matched: ${got.mkString(",")}")
  }

  private def loadVitInitialState(stateMem: AxiMemorySim, l: Int): Unit =
    writeI32Values(stateMem, STATE_VIT, readVitWindow(s"${vitBinaryDir(l)}/MHA_LN_X.bin"))

  private def compareVitState(stateMem: AxiMemorySim, expected: Array[Long], info: String): Unit = {
    val got = readI32Values(stateMem, STATE_VIT, VIT_NUM_X)
    var mismatch = 0
    for (i <- got.indices) {
      if (got(i) != expected(i)) {
        mismatch += 1
        if (mismatch <= 10) println(s"$info mismatch at $i: got=${got(i)} expected=${expected(i)}")
      }
    }
    require(mismatch == 0, s"$info has $mismatch mismatches")
    println(s"$info matched ($VIT_NUM_X int32 values)")
  }

  private def compareVitLayerOutput(stateMem: AxiMemorySim, l: Int): Unit =
    compareVitState(stateMem, readValues(s"${vitLayerDir(l)}/CONDENSED_RESIDUAL_Y.bin", I64), s"ACCELERATOR ViT layer $l state")

  private def writeCommonConfig(drv: AddressSeparableAxiLite4Driver): Unit = {
    drv.write(ctrl_cfg.ADDR_MODE, 0)
    drv.write(ctrl_cfg.ADDR_POS, TOP_POS)
    drv.write(ctrl_cfg.ADDR_CACHE_UPDATE_MODE, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_LLM_LNW, PARAM_LLM_LNW)
    drv.write(ctrl_cfg.ADDR_MEMORY_CLS_LNW, PARAM_CLS_LNW)
    drv.write(ctrl_cfg.ADDR_MEMORY_DECODER_W_LO, WEIGHT_DEC)
    drv.write(ctrl_cfg.ADDR_MEMORY_DECODER_W_HI, WEIGHT_DEC)
    drv.write(ctrl_cfg.ADDR_MEMORY_CLS_W_LO, WEIGHT_CLS)
    drv.write(ctrl_cfg.ADDR_MEMORY_CLS_W_HI, WEIGHT_CLS)
    drv.write(ctrl_cfg.ADDR_MEMORY_DECODER_STATE, STATE_DEC)
    drv.write(ctrl_cfg.ADDR_MEMORY_CLS_Y, STATE_CLS_Y)
    drv.write(ctrl_cfg.ADDR_MEMORY_K_CACHE, KV_BASE)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_BIAS, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_LNW, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_LNB, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_STATE, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_W_LO, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_W_HI, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_A, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_XM, 0)
  }

  private def writeVitConfig(drv: AddressSeparableAxiLite4Driver): Unit = {
    drv.write(ctrl_cfg.ADDR_MODE, 1)
    drv.write(ctrl_cfg.ADDR_POS, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_LLM_LNW, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_CLS_LNW, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_DECODER_W_LO, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_DECODER_W_HI, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_CLS_W_LO, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_CLS_W_HI, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_DECODER_STATE, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_CLS_Y, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_K_CACHE, 0)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_BIAS, PARAM_VIT_BIAS)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_LNW, PARAM_VIT_LNW)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_LNB, PARAM_VIT_LNB)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_STATE, STATE_VIT)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_W_LO, WEIGHT_VIT)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_W_HI, WEIGHT_VIT)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_A, VIT_A)
    drv.write(ctrl_cfg.ADDR_MEMORY_VIT_XM, VIT_XM)
  }

  private def launch(drv: AddressSeparableAxiLite4Driver, dut: ACCELERATOR, l: Int, runMask: BigInt, paramOp: Int, stateOp: Int): Unit = {
    drv.write(ctrl_cfg.ADDR_L_BEGIN, l)
    drv.write(ctrl_cfg.ADDR_L_CLOSE, l + 1)
    drv.write(ctrl_cfg.ADDR_PARAM_OP, paramOp)
    drv.write(ctrl_cfg.ADDR_STATE_OP, stateOp)
    drv.write(ctrl_cfg.ADDR_RUN_MASK, runMask)
    dut.clockDomain.waitSampling(20)
    drv.write(ctrl_cfg.ADDR_T, 1)
  }

  private def waitAllIdle(dut: ACCELERATOR, info: String): Unit = {
    var cycles = 0
    while (dut.io.idle.toBoolean && cycles < timeoutCycles) {
      dut.clockDomain.waitSampling()
      cycles += 1
    }
    require(!dut.io.idle.toBoolean, s"$info did not leave idle within $timeoutCycles cycles")
    cycles = 0
    while (!dut.io.idle.toBoolean && cycles < timeoutCycles) {
      dut.clockDomain.waitSampling()
      cycles += 1
      if (heartbeatCycles > 0 && cycles % heartbeatCycles == 0) {
        // 顶层 decoder layer 在 Verilator 下可能很慢；heartbeat 用于区分长跑和无进展等待。
        println(s"$info busy heartbeat: $cycles cycles")
      }
    }
    require(dut.io.idle.toBoolean, s"$info did not return idle within $timeoutCycles cycles")
    println(s"$info finished after $cycles busy cycles")
  }

  private case class AxisStat(name: String, stream: Axi4Stream) {
    var handshakes = 0L
    var validOnly = 0L
    var readyOnly = 0L
    private var lastHandshakes = 0L
    private var lastValidOnly = 0L
    private var lastReadyOnly = 0L

    def sample(): Unit = {
      val valid = stream.valid.toBoolean
      val ready = stream.ready.toBoolean
      if (valid && ready) handshakes += 1
      else if (valid) validOnly += 1
      else if (ready) readyOnly += 1
    }

    def changed: Boolean =
      handshakes != lastHandshakes || validOnly != lastValidOnly || readyOnly != lastReadyOnly

    def markPrinted(): Unit = {
      lastHandshakes = handshakes
      lastValidOnly = validOnly
      lastReadyOnly = readyOnly
    }
  }

  private case class AxisValueCheck(name: String, stream: Axi4Stream, expected: Array[Long], lanes: Int, signed: Boolean) {
    private val laneBits = stream.config.dataWidth * 8 / lanes
    private val beats = expected.length / lanes
    @volatile private var seen = 0
    @volatile private var mismatch = 0

    require(expected.length % lanes == 0, s"$name length ${expected.length} is not divisible by lanes=$lanes")

    def start(clockDomain: ClockDomain): SimThread = fork {
      // 顶层数值 tap 只观察已发生的握手，不驱动 ready/valid，用于定位闭合链第一个错流。
      while (seen < beats) {
        clockDomain.waitSamplingWhere(stream.valid.toBoolean && stream.ready.toBoolean)
        val tile = stream.data.toBigInt
        val base = seen * lanes
        for (lane <- 0 until lanes) {
          val gotBits = (tile >> (laneBits * lane)) & ((BigInt(1) << laneBits) - 1)
          val got = if (signed) signExtend(gotBits, laneBits) else gotBits.toLong
          val exp = normalizedExpected(expected(base + lane), laneBits, signed)
          if (got != exp) {
            mismatch += 1
            if (mismatch <= 10) {
              println(s"$name mismatch at ${base + lane}: got=$got expected=$exp laneBits=$laneBits")
            }
          }
        }
        seen += 1
      }
      require(mismatch == 0, s"$name has $mismatch mismatches")
      println(s"$name matched (${expected.length} elements, $lanes lanes)")
    }
  }

  private def startAcceleratorStreamChecks(dut: ACCELERATOR, l: Int): Seq[SimThread] = {
    if (!checkStreams) return Seq.empty
    val dir = layerDir(l)
    val bin = binaryDir(l)
    val mhaORes = readTokenWindow(s"$bin/MHA_O_RES.bin", POS, LLAMA_T, LLAMA_C)
    val mlpXdRes = readTokenWindow(s"$bin/MLP_XD_RES.bin", POS, LLAMA_T, LLAMA_C)
    val checks = Seq(
      AxisValueCheck(s"ACCELERATOR mux.q layer $l", dut.inst_mux.io.q_stream, readValues(s"$dir/CONDENSED_GEMM_X_Q.bin", I8), 64, signed = true),
      AxisValueCheck(s"ACCELERATOR mux.s layer $l", dut.inst_mux.io.s_stream, readValues(s"$dir/CONDENSED_GEMM_X_S.bin", I8), 8, signed = false),
      AxisValueCheck(s"ACCELERATOR weight.wq layer $l", dut.inst_weight_axi.io.wq_stream, readValues(s"$dir/CONDENSED_GEMM_W_Q.bin", I8), 64, signed = true),
      AxisValueCheck(s"ACCELERATOR weight.ws1 layer $l", dut.inst_weight_axi.io.ws1_stream, readValues(s"$dir/CONDENSED_GEMM_W_S1.bin", I8), 8, signed = false),
      AxisValueCheck(s"ACCELERATOR weight.ws2 layer $l", dut.inst_weight_axi.io.ws2_stream, readValues(s"$dir/CONDENSED_GEMM_W_S2.bin", I8), 8, signed = false),
      AxisValueCheck(s"ACCELERATOR permute.o layer $l", dut.inst_permute.io.o_stream, readValues(s"$dir/CONDENSED_GEMM_Y.bin", I64), 8, signed = true),
      AxisValueCheck(s"ACCELERATOR demux.od layer $l", dut.inst_demux.io.od_fc2_stream, readValues(s"$dir/CONDENSED_DEMUX_OD.bin", I64), 8, signed = true),
      AxisValueCheck(s"ACCELERATOR res.o layer $l", dut.inst_residual.io.res_o_stream, readValues(s"$dir/CONDENSED_RESIDUAL_O.bin", I64), 8, signed = true),
      AxisValueCheck(s"ACCELERATOR res.y layer $l", dut.inst_residual.io.y_stream, llmDeltaOrder(mhaORes) ++ llmDeltaOrder(mlpXdRes), 8, signed = true)
    )
    checks.map(_.start(dut.clockDomain))
  }

  private def startAcceleratorClsStreamChecks(dut: ACCELERATOR, clsInputTokenStart: Int): Seq[SimThread] = {
    if (!checkStreams) return Seq.empty
    val clsDir = layerDir(LLAMA_L)
    val resExpected = clsStateOverride match {
      case Some(_) => clsInputState()
      case None    => readTokenWindow(s"${binaryDir(LLAMA_L)}/CLS_X.bin", clsInputTokenStart, LLAMA_T, LLAMA_C)
    }
    val fullChecks = Seq(
      // CLS/lm_head 主链复用 RESIDUAL 做 state replay；这里先确认顶层喂给 final RMSNorm 的窗口语义。
      AxisValueCheck("ACCELERATOR cls res.o", dut.inst_residual.io.res_o_stream, resExpected, 8, signed = true),
      AxisValueCheck("ACCELERATOR cls rms.xlnq", dut.inst_rms_layernorm.io.xlnq_stream, readValues(clsDebugFile("CONDENSED_XLN_Q.bin"), I8), 8, signed = true),
      AxisValueCheck("ACCELERATOR cls rms.xlns", dut.inst_rms_layernorm.io.xlns_stream, readValues(clsDebugFile("CONDENSED_XLN_S.bin"), I8), 1, signed = false),
      AxisValueCheck("ACCELERATOR cls mux.q", dut.inst_mux.io.q_stream, readValues(clsDebugFile("CONDENSED_GEMM_X_Q.bin"), I8), 64, signed = true),
      AxisValueCheck("ACCELERATOR cls mux.s", dut.inst_mux.io.s_stream, readValues(clsDebugFile("CONDENSED_GEMM_X_S.bin"), I8), 8, signed = false),
      AxisValueCheck("ACCELERATOR cls weight.wq", dut.inst_weight_axi.io.wq_stream, readValues(clsDebugFile("CONDENSED_GEMM_W_Q.bin"), I8), 64, signed = true),
      AxisValueCheck("ACCELERATOR cls weight.ws1", dut.inst_weight_axi.io.ws1_stream, readValues(clsDebugFile("CONDENSED_GEMM_W_S1.bin"), I8), 8, signed = false),
      AxisValueCheck("ACCELERATOR cls weight.ws2", dut.inst_weight_axi.io.ws2_stream, readValues(clsDebugFile("CONDENSED_GEMM_W_S2.bin"), I8), 8, signed = false),
      AxisValueCheck("ACCELERATOR cls permute.o", dut.inst_permute.io.o_stream, readValues(clsDebugFile("CONDENSED_GEMM_Y.bin"), I64), 8, signed = true),
      AxisValueCheck("ACCELERATOR cls demux.cls", dut.inst_demux.io.cls_stream, clsExpectedIndex(), 8, signed = true)
    )
    val checks =
      if ((clsStateOverride.isDefined || clsExpectedOverride.isDefined) && clsDebugDirOverride.isEmpty) {
        // 自定义 state/expected 时，默认只检查真实输入窗口和最终 index；
        // 中间 XLN/MUX/PERMUTE 参考需要另外生成，不能误用 POS=96 的全局 condensed 文件。
        Seq(
          AxisValueCheck("ACCELERATOR cls res.o", dut.inst_residual.io.res_o_stream, resExpected, 8, signed = true),
          AxisValueCheck("ACCELERATOR cls demux.cls", dut.inst_demux.io.cls_stream, clsExpectedIndex(), 8, signed = true)
        )
      } else {
        fullChecks
      }
    checks.map(_.start(dut.clockDomain))
  }

  private def startAcceleratorDebugMonitors(dut: ACCELERATOR): Unit = {
    // 顶层 decoder layer 长跑时使用：统计闭合链关键 AXIS 握手和各 IP idle 状态，
    // 不改变硬件 IO，只在 VLM_ACCEL_DEBUG=1 的仿真中打印反压位置。
    if (!debugInternal) return
    val period = if (heartbeatCycles > 0) heartbeatCycles else 1000000
    val stats = Seq(
      AxisStat("m_axi.lnw",        dut.inst_m_axi.io.lnw_stream),
      AxisStat("weight.wq",        dut.inst_weight_axi.io.wq_stream),
      AxisStat("weight.ws1",       dut.inst_weight_axi.io.ws1_stream),
      AxisStat("weight.ws2",       dut.inst_weight_axi.io.ws2_stream),
      AxisStat("kv.kq_i",          dut.inst_kv_cache.io.kq_cache_i_stream),
      AxisStat("kv.ks_i",          dut.inst_kv_cache.io.ks_cache_i_stream),
      AxisStat("kv.vq_i",          dut.inst_kv_cache.io.vq_cache_i_stream),
      AxisStat("kv.vs_i",          dut.inst_kv_cache.io.vs_cache_i_stream),
      AxisStat("qk.kq_o",          dut.inst_qk_gemm.io.kq_cache_o_stream),
      AxisStat("qk.ks_o",          dut.inst_qk_gemm.io.ks_cache_o_stream),
      AxisStat("rv.vq_o",          dut.inst_rv_gemm.io.vq_cache_o_stream),
      AxisStat("rv.vs_o",          dut.inst_rv_gemm.io.vs_cache_o_stream),
      AxisStat("state.x",          dut.inst_state_axi.io.x_stream),
      AxisStat("res.y",            dut.inst_residual.io.y_stream),
      AxisStat("demux.cls",        dut.inst_demux.io.cls_stream),
      AxisStat("rms.xlnq",         dut.inst_rms_layernorm.io.xlnq_stream),
      AxisStat("rms.xlns",         dut.inst_rms_layernorm.io.xlns_stream),
      AxisStat("mux.q",            dut.inst_mux.io.q_stream),
      AxisStat("mux.s",            dut.inst_mux.io.s_stream),
      AxisStat("permute.o",        dut.inst_permute.io.o_stream),
      AxisStat("demux.qk",         dut.inst_demux.io.qk_stream),
      AxisStat("demux.v",          dut.inst_demux.io.v_stream),
      AxisStat("demux.ug",         dut.inst_demux.io.mlp1_stream),
      AxisStat("demux.od",         dut.inst_demux.io.od_fc2_stream),
      AxisStat("rope.q",           dut.inst_rope_qk.io.qk_q_stream),
      AxisStat("rope.s",           dut.inst_rope_qk.io.qk_s_stream),
      AxisStat("qk.r",             dut.inst_qk_gemm.io.r_stream),
      AxisStat("softmax.rq",       dut.inst_softmax.io.rq_stream),
      AxisStat("softmax.rs",       dut.inst_softmax.io.rs_stream),
      AxisStat("rv.aq",            dut.inst_rv_gemm.io.aq_stream),
      AxisStat("rv.as",            dut.inst_rv_gemm.io.as_stream),
      AxisStat("areorder.aq",      dut.inst_a_reorder.io.mux_aq_stream),
      AxisStat("areorder.as",      dut.inst_a_reorder.io.mux_as_stream),
      AxisStat("silu.q",           dut.inst_silu_gelu.io.q_stream),
      AxisStat("silu.s",           dut.inst_silu_gelu.io.s_stream),
      AxisStat("xmreorder.q",      dut.inst_xm_reorder.io.mux_q_stream),
      AxisStat("xmreorder.s",      dut.inst_xm_reorder.io.mux_s_stream),
      AxisStat("res.o",            dut.inst_residual.io.res_o_stream)
    )
    val ipIdle = Seq(
      "M_AXI" -> dut.inst_m_axi.io.idle,
      "WEIGHT_AXI" -> dut.inst_weight_axi.io.idle,
      "KV_CACHE" -> dut.inst_kv_cache.io.idle,
      "STATE_AXI" -> dut.inst_state_axi.io.idle,
      "MUX" -> dut.inst_mux.io.idle,
      "PERMUTE" -> dut.inst_permute.io.idle,
      "DEMUX" -> dut.inst_demux.io.idle,
      "ROPE_QK" -> dut.inst_rope_qk.io.idle,
      "QK_GEMM" -> dut.inst_qk_gemm.io.idle,
      "SOFTMAX" -> dut.inst_softmax.io.idle,
      "RV_GEMM" -> dut.inst_rv_gemm.io.idle,
      "A_REORDER" -> dut.inst_a_reorder.io.idle,
      "SILU_GELU" -> dut.inst_silu_gelu.io.idle,
      "XM_REORDER" -> dut.inst_xm_reorder.io.idle,
      "RESIDUAL" -> dut.inst_residual.io.idle,
      "RMS_LAYERNORM" -> dut.inst_rms_layernorm.io.idle
    )
    fork {
      var cycles = 0L
      while (true) {
        dut.clockDomain.waitSampling()
        cycles += 1
        stats.foreach(_.sample())
        if (cycles % period == 0) {
          val active = ipIdle.filterNot(_._2.toBoolean).map(_._1).mkString(",")
          println(s"ACCELERATOR debug cycles=$cycles active=${if (active.nonEmpty) active else "none"}")
          stats.filter(s => s.handshakes > 0 || s.validOnly > 0 || s.readyOnly > 0 || s.changed).foreach { s =>
            println(s"  ${s.name}: hs=${s.handshakes} validOnly=${s.validOnly} readyOnly=${s.readyOnly}")
            s.markPrinted()
          }
        }
      }
    }
  }

  private def runDecoderLayer(drv: AddressSeparableAxiLite4Driver, dut: ACCELERATOR, stateMem: AxiMemorySim, l: Int): Unit = {
    println(s"Launching ACCELERATOR decoder layer $l, pos=$TOP_POS, chunk=$CHUNK")
    val streamChecks = startAcceleratorStreamChecks(dut, l)
    launch(drv, dut, l, decoderMask, paramOp = 1, stateOp = STATE_LAYER_OP)
    waitAllIdle(dut, s"ACCELERATOR decoder layer $l")
    streamChecks.foreach(_.join())
    compareLayerOutput(stateMem, l)
  }

  private def runCls(drv: AddressSeparableAxiLite4Driver, dut: ACCELERATOR, stateMem: AxiMemorySim, clsInputTokenStart: Int): Unit = {
    println("Launching ACCELERATOR CLS/lm_head main chain")
    writeI32Values(stateMem, STATE_CLS_Y, Array.fill[Long](LLAMA_T)(0))
    val streamChecks = startAcceleratorClsStreamChecks(dut, clsInputTokenStart)
    launch(drv, dut, LLAMA_L, clsMainMask, paramOp = 1, stateOp = 0)
    dut.clockDomain.waitSampling(clsWriteDelay)
    println("Launching ACCELERATOR CLS index writeback")
    launch(drv, dut, LLAMA_L, mask(ctrl_cfg.IP_STATE_AXI), paramOp = 1, stateOp = 3)
    waitAllIdle(dut, "ACCELERATOR CLS/lm_head")
    streamChecks.foreach(_.join())
    compareClsOutput(stateMem)
  }

  private def runVitLayer(drv: AddressSeparableAxiLite4Driver, dut: ACCELERATOR, stateMem: AxiMemorySim, l: Int): Unit = {
    // 同一套 M_AXI 先预取 ViT LayerNorm gamma/beta 到 256-depth FIFO；
    // 主链再启动 M_AXI bias op，与 DEMUX 消费同步推进。
    println(s"Preloading ACCELERATOR ViT layer $l norm params")
    launch(drv, dut, l, mask(ctrl_cfg.IP_M_AXI), paramOp = 1, stateOp = 0)
    waitAllIdle(dut, s"ACCELERATOR ViT layer $l norm preload")

    println(s"Launching ACCELERATOR ViT layer $l main chain")
    launch(drv, dut, l, vitMainMask, paramOp = 0, stateOp = STATE_LAYER_OP)
    waitAllIdle(dut, s"ACCELERATOR ViT layer $l")
    compareVitLayerOutput(stateMem, l)
  }

  private def parseLayerList(raw: String): Seq[Int] =
    raw.split("[,\\s]+").iterator.filter(_.nonEmpty).map(_.toInt).toSeq

  def run(mode: String): Unit = {
    val selectedLayers = parseLayerList(sys.env.getOrElse("VLM_ACCEL_LAYERS", "0"))
    simConfig.compile(new ACCELERATOR).doSimUntilVoid { dut =>
      val drv = AddressSeparableAxiLite4Driver(dut.io.axilite, dut.clockDomain)
      val axiCfg = AxiMemorySimConfig(
        maxOutstandingReads = 32,
        maxOutstandingWrites = 16,
        readResponseDelay = 0,
        writeResponseDelay = 0,
        interruptProbability = 0
      )
      val paramMem = AxiMemorySim(dut.io.param_gmem, dut.clockDomain, axiCfg)
      val weightLo = AxiMemorySim(dut.io.weight_gmem1, dut.clockDomain, axiCfg)
      val weightHi = AxiMemorySim(dut.io.weight_gmem2, dut.clockDomain, axiCfg)
      val stateMem = AxiMemorySim(dut.io.state_gmem, dut.clockDomain, axiCfg)
      val vitAMem = AxiMemorySim(dut.io.vit_a_gmem, dut.clockDomain, axiCfg)
      val vitXMMem = AxiMemorySim(dut.io.vit_xm_gmem, dut.clockDomain, axiCfg)
      val kvMem = AxiMemorySim(dut.io.kv_gmem, dut.clockDomain, axiCfg)
      Seq(paramMem, weightLo, weightHi, stateMem, vitAMem, vitXMMem, kvMem).foreach(_.reset())
      drv.reset()
      init_clock(dut.clockDomain, 10)
      writeCommonConfig(drv)
      writeNormParams(paramMem)
      startAcceleratorDebugMonitors(dut)

      mode match {
        case "layers" =>
          selectedLayers.foreach { l =>
            require(l >= 0 && l < LLAMA_L, s"Invalid decoder layer: $l")
            loadInitialState(stateMem, l)
            loadLayerInputs(weightLo, weightHi, kvMem, l)
            runDecoderLayer(drv, dut, stateMem, l)
          }
        case "cls" =>
          loadClsInitialState(stateMem)
          loadClsWeight(weightLo, weightHi)
          runCls(drv, dut, stateMem, clsInputTokenStart = POS)
        case "decoder" | "decoder_cls" =>
          loadInitialState(stateMem, 0)
          for (l <- 0 until LLAMA_L) {
            loadLayerInputs(weightLo, weightHi, kvMem, l)
            runDecoderLayer(drv, dut, stateMem, l)
            if (l + 1 < LLAMA_L) {
              compareState(stateMem, readTokenWindow(s"${binaryDir(l + 1)}/MHA_X.bin", POS, LLAMA_T, LLAMA_C), s"ACCELERATOR layer ${l + 1} input carry")
            }
          }
          if (mode == "decoder_cls") {
            compareClsInput(stateMem)
            loadClsWeight(weightLo, weightHi)
            runCls(drv, dut, stateMem, clsInputTokenStart = POS)
          }
        case "last_decoder_cls" =>
          // 快速覆盖最终链路：只从最后一层 decoder 的 MHA_X 初始化 state，
          // 跑 layer 31 写回，再接 final RMSNorm/CLS/lm_head 写回 argmax index。
          val last = LLAMA_L - 1
          loadInitialState(stateMem, last)
          loadLayerInputs(weightLo, weightHi, kvMem, last)
          runDecoderLayer(drv, dut, stateMem, last)
          compareClsInput(stateMem)
          loadClsWeight(weightLo, weightHi)
          runCls(drv, dut, stateMem, clsInputTokenStart = POS)
        case "vit_layer" =>
          writeVitConfig(drv)
          writeVitParams(paramMem)
          writeVitPackedWeights(weightLo, weightHi)
          val l = sys.env.getOrElse("VLM_VIT_ACCEL_LAYER", "0").toInt
          require(l >= 0 && l < VIT_L, s"Invalid ViT layer: $l")
          loadVitInitialState(stateMem, l)
          runVitLayer(drv, dut, stateMem, l)
        case "vit_2layers" =>
          writeVitConfig(drv)
          writeVitParams(paramMem)
          writeVitPackedWeights(weightLo, weightHi)
          loadVitInitialState(stateMem, 0)
          runVitLayer(drv, dut, stateMem, 0)
          compareVitState(stateMem, readVitWindow(s"${vitBinaryDir(1)}/MHA_LN_X.bin"), "ACCELERATOR ViT layer 1 input carry")
          runVitLayer(drv, dut, stateMem, 1)
        case other =>
          throw new IllegalArgumentException(s"Unknown accelerator payload mode: $other")
      }
      simSuccess()
    }
  }
}

object simulate_accelerator_reuse extends App {
  val mode = if (args.nonEmpty) args(0) else sys.env.getOrElse("VLM_ACCEL_TEST", "layers")
  AcceleratorReusePayloadSim.run(mode)
}

object simulate_accelerator_llm_layers extends App { AcceleratorReusePayloadSim.run("layers") }
object simulate_accelerator_llm_cls extends App { AcceleratorReusePayloadSim.run("cls") }
object simulate_accelerator_llm_decoder extends App { AcceleratorReusePayloadSim.run("decoder") }
object simulate_accelerator_llm_decoder_cls extends App { AcceleratorReusePayloadSim.run("decoder_cls") }
object simulate_accelerator_llm_last_decoder_cls extends App { AcceleratorReusePayloadSim.run("last_decoder_cls") }
object simulate_accelerator_vit_layer extends App { AcceleratorReusePayloadSim.run("vit_layer") }
object simulate_accelerator_vit_2layers extends App { AcceleratorReusePayloadSim.run("vit_2layers") }

// @formatter:on
