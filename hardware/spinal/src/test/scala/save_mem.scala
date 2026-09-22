import utils._

import java.io.{File, FileOutputStream}
import scala.language.postfixOps

// @formatter:off
object save_mem extends App {
  // 复用版权重预处理：把 HLS step1 生成的 W_Q/W_S1/W_S2 重新打成 WEIGHT_AXI 读取的 CONDENSED_WEIGHT。
  // 默认同时处理 LLM decoder、LLM CLS/lm_head 和 ViT encoder；可传入 llm/cls/vit 只处理部分。

  val LLAMA_L = 32
  val LLAMA_C = 960
  val LLAMA_CM = 2560
  val LLAMA_VOCAB = 49280

  val VIT_L = 12
  val VIT_C = 768
  val VIT_CM = 3072

  val G = 8
  val DW_WQ = 5
  val DW_WS = 4
  val DW_MAXI = 256
  val BYTE_PER_PACK = DW_MAXI / 8

  // 与 src/reuse_axi_common.h 中的 AXI_WQ_CYCS/AXI_WS_CYCS 保持一致。
  def gcd(a: Int, b: Int): Int = if (b == 0) math.abs(a) else gcd(b, a % b)
  def lcm(a: Int, b: Int): Int = a / gcd(a, b) * b
  val WQ_UNIT_BITS = DW_WQ * G * G
  val WS_UNIT_BITS = DW_WS * 2 * G
  val WQ_WS_GCD = gcd(WQ_UNIT_BITS, WS_UNIT_BITS)
  val WQ_RATIO = WQ_UNIT_BITS / WQ_WS_GCD
  val WS_RATIO = WS_UNIT_BITS / WQ_WS_GCD
  val WQ_CYCS_AGGR_SIZE = lcm(DW_MAXI, WQ_UNIT_BITS) / DW_MAXI
  val WS_CYCS_AGGR_SIZE = lcm(DW_MAXI, WS_UNIT_BITS) / DW_MAXI
  val K_WQ = WQ_CYCS_AGGR_SIZE / gcd(WQ_RATIO, WQ_CYCS_AGGR_SIZE)
  val K_WS = WS_CYCS_AGGR_SIZE / gcd(WS_RATIO, WS_CYCS_AGGR_SIZE)
  val K_SUPER = lcm(K_WQ, K_WS)
  val WQ_CYCS = WQ_RATIO * K_SUPER
  val WS_CYCS = WS_RATIO * K_SUPER
  val WQ_PACK_PER_LOOP = WQ_CYCS * DW_MAXI / (DW_WQ * G * G)
  val WS_PACK_PER_LOOP = WS_CYCS * DW_MAXI / (DW_WS * 2 * G)

  case class WeightShape(numWq: Int, numWs: Int) {
    val bytes: Int = (numWq * DW_WQ + numWs * DW_WS * 2) / 8
    val cycs: Int = bytes / BYTE_PER_PACK
    val loops: Int = cycs / (WQ_CYCS + WS_CYCS)
  }

  val decoderShape = WeightShape(
    numWq = 4 * LLAMA_C * LLAMA_C + 3 * LLAMA_C * LLAMA_CM,
    numWs = (4 * LLAMA_C * LLAMA_C + 3 * LLAMA_C * LLAMA_CM) / G
  )
  val clsShape = WeightShape(
    numWq = LLAMA_VOCAB * LLAMA_C,
    numWs = LLAMA_VOCAB * LLAMA_C / G
  )
  val vitShape = WeightShape(
    numWq = 4 * VIT_C * VIT_C + 2 * VIT_C * VIT_CM,
    numWs = (4 * VIT_C * VIT_C + 2 * VIT_C * VIT_CM) / G
  )

  def selected(name: String): Boolean = args.isEmpty || args.exists(_.equalsIgnoreCase(name))

  def ensureParent(path: String): Unit = {
    val parent = new File(path).getParentFile
    if (parent != null) parent.mkdirs()
  }

  def composeLayer(
                    layerName: String,
                    layerDir: String,
                    shape: WeightShape
                  ): Unit = {
    val wqPath = s"$layerDir/CONDENSED_GEMM_W_Q.bin"
    val ws1Path = s"$layerDir/CONDENSED_GEMM_W_S1.bin"
    val ws2Path = s"$layerDir/CONDENSED_GEMM_W_S2.bin"
    val outPath = s"$layerDir/CONDENSED_WEIGHT.bin"

    println(s"Composing $layerName")
    val refWq = read_int8_file(wqPath, shape.numWq)
    val refWs1 = read_int8_file(ws1Path, shape.numWs)
    val refWs2 = read_int8_file(ws2Path, shape.numWs)
    val condensed = Array.ofDim[Byte](shape.cycs * BYTE_PER_PACK)

    for (loop <- 0 until shape.loops) {
      val loopWq = new Array[BigInt](WQ_PACK_PER_LOOP * G * G)
      val loopWs = new Array[BigInt](WS_PACK_PER_LOOP * G * 2)

      for (idx <- 0 until WQ_PACK_PER_LOOP * G * G) {
        loopWq(idx) = BigInt(refWq(loop * WQ_PACK_PER_LOOP * G * G + idx))
      }
      for (idx <- 0 until WS_PACK_PER_LOOP * G) {
        loopWs(idx * 2 + 0) = BigInt(refWs1(loop * WS_PACK_PER_LOOP * G + idx))
        loopWs(idx * 2 + 1) = BigInt(refWs2(loop * WS_PACK_PER_LOOP * G + idx))
      }

      val wqBytes = resolve_tile(compose_tile(loopWq, DW_WQ), 8, WQ_CYCS * BYTE_PER_PACK, is_signed = false).map(_.toByte)
      val wsBytes = resolve_tile(compose_tile(loopWs, DW_WS), 8, WS_CYCS * BYTE_PER_PACK, is_signed = false).map(_.toByte)
      val base = loop * (WQ_CYCS + WS_CYCS) * BYTE_PER_PACK

      for (idx <- wqBytes.indices) condensed(base + idx) = wqBytes(idx)
      for (idx <- wsBytes.indices) condensed(base + WQ_CYCS * BYTE_PER_PACK + idx) = wsBytes(idx)
    }

    ensureParent(outPath)
    val file = new FileOutputStream(outPath)
    try file.write(condensed) finally file.close()
    println(s"Writing $outPath done")
  }

  if (selected("llm")) {
    (0 until LLAMA_L).foreach { l =>
      composeLayer(s"LLM decoder_$l", s"${ctrl_cfg.condense_prefix}$l", decoderShape)
    }
  }

  if (selected("cls")) {
    composeLayer(s"LLM cls/lm_head_$LLAMA_L", s"${ctrl_cfg.condense_prefix}$LLAMA_L", clsShape)
  }

  if (selected("vit")) {
    (0 until VIT_L).foreach { l =>
      composeLayer(s"ViT vision_$l", s"${ctrl_cfg.vit_condense_prefix}$l", vitShape)
    }
  }
}
