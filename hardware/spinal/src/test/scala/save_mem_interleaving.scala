import utils._

import java.io.{File, FileOutputStream}
import java.nio.{ByteBuffer, ByteOrder}
import scala.collection.mutable.ArrayBuffer

// @formatter:off
object save_mem_interleaving extends App {
  // 复用版权重内存布局生成：
  //   输入：每层 CONDENSED_WEIGHT.bin，内部仍是每个 256-bit 周期按 LO/Hl 两个 128-bit 半拍交织。
  //   输出：all_*_w.bin，布局为 all_layer_lo_half ++ all_layer_hi_half，匹配 WEIGHT_AXI 的 LO/HI 两个 M_AXI bundle。
  //   LLM norm 参数额外从 src/ref/LLM 导出为 int32 little-endian 连续文件，匹配 M_AXI 的 lnw DDR 布局。

  val LLAMA_L = 32
  val LLAMA_C = 960
  val LLAMA_H = 15
  val LLAMA_KVH = 5
  val LLAMA_HC = 64
  val LLAMA_GQA = LLAMA_H / LLAMA_KVH
  val LLAMA_CM = 2560
  val LLAMA_VOCAB = 49280

  val VIT_L = 12
  val VIT_C = 768
  val VIT_HC = 64
  val VIT_CM = 3072

  val G = 8
  val DW_WQ = 5
  val DW_WS = 4
  val DW_MAXI = 256
  val BYTE_PER_MAXI = DW_MAXI / 8
  val BYTE_PER_HALF = BYTE_PER_MAXI / 2
  val LLM_REF_PATH = sys.env.getOrElse("VLM_LLM_REF_PATH", "../src/ref/LLM")
  val VIT_REF_PATH = sys.env.getOrElse("VLM_VIT_REF_PATH", "../src/ref/ViT")
  val VIT_BIN_PREFIX = sys.env.getOrElse("VLM_VIT_BIN_PREFIX", "../Weights/ViT/binaries/vision_")

  // 与 save_mem.scala / src/reuse_axi_common.h 保持一致：WQ 与 WS 先按 bit 比例合成同一个 256-bit 超周期。
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
    val bytesHalf: Int = bytes / 2
    val cycs: Int = bytes / BYTE_PER_MAXI
  }

  val decoderNumWq = (LLAMA_H + 2 * LLAMA_KVH) * LLAMA_HC * LLAMA_C + LLAMA_C * LLAMA_C + 2 * LLAMA_CM * LLAMA_C + LLAMA_C * LLAMA_CM
  val decoderShape = WeightShape(
    numWq = decoderNumWq,
    numWs = decoderNumWq / G
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

  def ensureDir(path: String): Unit = new File(path).mkdirs()

  def splitLoHi(condensed: Array[Byte], shape: WeightShape): (Array[Byte], Array[Byte]) = {
    val lo = Array.ofDim[Byte](shape.bytesHalf)
    val hi = Array.ofDim[Byte](shape.bytesHalf)
    for (cyc <- 0 until shape.cycs; loHi <- 0 until 2; n <- 0 until BYTE_PER_HALF) {
      val wByte = condensed(cyc * BYTE_PER_MAXI + loHi * BYTE_PER_HALF + n)
      val wAddr = cyc * BYTE_PER_HALF + n
      if (loHi == 0) lo(wAddr) = wByte else hi(wAddr) = wByte
    }
    (lo, hi)
  }

  def writeFile(path: String, data: Array[Byte]): Unit = {
    val parent = new File(path).getParentFile
    if (parent != null) parent.mkdirs()
    val file = new FileOutputStream(path)
    try file.write(data) finally file.close()
    println(s"Saved $path (${data.length} bytes)")
  }

  def readTextInts(path: String): Array[Int] = {
    val source = scala.io.Source.fromFile(path)
    val text = try source.mkString finally source.close()
    text.split("[,;\\s]+").iterator.filter(_.nonEmpty).map(_.toInt).toArray
  }

  def readTextLongs(path: String): Array[Long] = {
    val source = scala.io.Source.fromFile(path)
    val text = try source.mkString finally source.close()
    text.split("[,;\\s]+").iterator.filter(_.nonEmpty).map(_.toLong).toArray
  }

  def readInt8MaybeBin(binPath: String, txtPath: String, expected: Int): Array[Byte] = {
    val bin = new File(binPath)
    if (bin.isFile) {
      require(bin.length() == expected, s"Unexpected $binPath size ${bin.length()}, expected $expected")
      read_int8_file(binPath, expected)
    } else {
      val values = readTextInts(txtPath)
      require(values.length == expected, s"Unexpected $txtPath length ${values.length}, expected $expected")
      values.map(_.toByte)
    }
  }

  def readI64MaybeBin(binPath: String, txtPath: String, expected: Int): Array[Long] = {
    val bin = new File(binPath)
    if (bin.isFile) {
      require(bin.length() == expected.toLong * 8, s"Unexpected $binPath size ${bin.length()}, expected ${expected * 8L}")
      read_int64_file(binPath, expected)
    } else {
      val values = readTextLongs(txtPath)
      require(values.length == expected, s"Unexpected $txtPath length ${values.length}, expected $expected")
      values
    }
  }

  def writeInt32File(path: String, data: Array[Int]): Unit = {
    // M_AXI 按 256-bit 一拍读取 8 个 int32；文件只需保持 int32 little-endian 连续排列。
    val buffer = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN)
    data.foreach(buffer.putInt)
    writeFile(path, buffer.array())
  }

  def writeInt64File(path: String, data: Array[Long]): Unit = {
    // ViT LayerNorm beta 为 int64；M_AXI 每 256-bit 读取 4 个，再组合成 CP=8。
    val buffer = ByteBuffer.allocate(data.length * 8).order(ByteOrder.LITTLE_ENDIAN)
    data.foreach(buffer.putLong)
    writeFile(path, buffer.array())
  }

  def collectLayeredWeights(
                             prefix: String,
                             layers: Int,
                             shape: WeightShape
                           ): Array[Byte] = {
    val allLo = Array.ofDim[Byte](layers * shape.bytesHalf)
    val allHi = Array.ofDim[Byte](layers * shape.bytesHalf)
    for (l <- 0 until layers) {
      val path = s"$prefix$l/CONDENSED_WEIGHT.bin"
      val (lo, hi) = splitLoHi(read_int8_file(path, shape.bytes), shape)
      Array.copy(lo, 0, allLo, l * shape.bytesHalf, shape.bytesHalf)
      Array.copy(hi, 0, allHi, l * shape.bytesHalf, shape.bytesHalf)
    }
    allLo ++ allHi
  }

  def collectSingleWeight(path: String, shape: WeightShape): Array[Byte] = {
    val (lo, hi) = splitLoHi(read_int8_file(path, shape.bytes), shape)
    lo ++ hi
  }

  def composePackedLayer(refWq: Array[Byte], refWs1: Array[Byte], refWs2: Array[Byte], shape: WeightShape): Array[Byte] = {
    require(refWq.length == shape.numWq, s"Unexpected WQ length ${refWq.length}, expected ${shape.numWq}")
    require(refWs1.length == shape.numWs, s"Unexpected WS1 length ${refWs1.length}, expected ${shape.numWs}")
    require(refWs2.length == shape.numWs, s"Unexpected WS2 length ${refWs2.length}, expected ${shape.numWs}")
    val condensed = Array.ofDim[Byte](shape.cycs * BYTE_PER_MAXI)

    for (loop <- 0 until shape.cycs / (WQ_CYCS + WS_CYCS)) {
      val loopWq = new Array[BigInt](WQ_PACK_PER_LOOP * G * G)
      val loopWs = new Array[BigInt](WS_PACK_PER_LOOP * G * 2)

      for (idx <- 0 until WQ_PACK_PER_LOOP * G * G) {
        loopWq(idx) = BigInt(refWq(loop * WQ_PACK_PER_LOOP * G * G + idx))
      }
      for (idx <- 0 until WS_PACK_PER_LOOP * G) {
        loopWs(idx * 2 + 0) = BigInt(refWs1(loop * WS_PACK_PER_LOOP * G + idx))
        loopWs(idx * 2 + 1) = BigInt(refWs2(loop * WS_PACK_PER_LOOP * G + idx))
      }

      val wqBytes = resolve_tile(compose_tile(loopWq, DW_WQ), 8, WQ_CYCS * BYTE_PER_MAXI, is_signed = false).map(_.toByte)
      val wsBytes = resolve_tile(compose_tile(loopWs, DW_WS), 8, WS_CYCS * BYTE_PER_MAXI, is_signed = false).map(_.toByte)
      val base = loop * (WQ_CYCS + WS_CYCS) * BYTE_PER_MAXI
      for (idx <- wqBytes.indices) condensed(base + idx) = wqBytes(idx)
      for (idx <- wsBytes.indices) condensed(base + WQ_CYCS * BYTE_PER_MAXI + idx) = wsBytes(idx)
    }
    condensed
  }

  def appendMatrixWeights(
                           qOut: ArrayBuffer[Byte],
                           s1Out: ArrayBuffer[Byte],
                           s2Out: ArrayBuffer[Byte],
                           q: Array[Byte],
                           s1: Array[Byte],
                           s2: Array[Byte],
                           rowOffset: Int,
                           rows: Int,
                           cols: Int
                         ): Unit = {
    // ViT WEIGHT_AXI 只保存 compact 权重；PERMUTE 的 TT 重放由调度端处理，不能把 128 份重复权重写入 DDR。
    val rowTiles = rows / G
    val colTiles = cols / G
    val scaleCols = cols / G
    for (rt <- 0 until rowTiles; ct <- 0 until colTiles) {
      for (tp <- 0 until G; cp <- 0 until G) {
        qOut += q((rowOffset + rt * G + tp) * cols + ct * G + cp)
      }
    }
    for (rt <- 0 until rowTiles; ct <- 0 until scaleCols; tp <- 0 until G) {
      s1Out += s1((rowOffset + rt * G + tp) * scaleCols + ct)
      s2Out += s2((rowOffset + rt * G + tp) * scaleCols + ct)
    }
  }


  def collectLlmRawLayer(l: Int): (Array[Byte], Array[Byte], Array[Byte]) = {
    val dir = s"${ctrl_cfg.condense_prefix}$l".replace("condense", "binaries")
    def i8(name: String, expected: Int): Array[Byte] =
      readInt8MaybeBin(s"$dir/$name.bin", s"$dir/$name.txt", expected)

    val qOut = ArrayBuffer[Byte]()
    val s1Out = ArrayBuffer[Byte]()
    val s2Out = ArrayBuffer[Byte]()

    val wqQ  = i8("MHA_WQ_Q",  LLAMA_C * LLAMA_C)
    val wqS1 = i8("MHA_WQ_S1", LLAMA_C * (LLAMA_C / G))
    val wqS2 = i8("MHA_WQ_S2", LLAMA_C * (LLAMA_C / G))
    val wkQ  = i8("MHA_WK_Q",  LLAMA_C * LLAMA_C)
    val wkS1 = i8("MHA_WK_S1", LLAMA_C * (LLAMA_C / G))
    val wkS2 = i8("MHA_WK_S2", LLAMA_C * (LLAMA_C / G))
    val wvQ  = i8("MHA_WV_Q",  LLAMA_C * LLAMA_C)
    val wvS1 = i8("MHA_WV_S1", LLAMA_C * (LLAMA_C / G))
    val wvS2 = i8("MHA_WV_S2", LLAMA_C * (LLAMA_C / G))

    for (kv <- 0 until LLAMA_KVH) {
      val replicatedRow = kv * LLAMA_GQA * LLAMA_HC
      appendMatrixWeights(qOut, s1Out, s2Out, wkQ, wkS1, wkS2, replicatedRow, LLAMA_HC, LLAMA_C)
      appendMatrixWeights(qOut, s1Out, s2Out, wvQ, wvS1, wvS2, replicatedRow, LLAMA_HC, LLAMA_C)
      for (r <- 0 until LLAMA_GQA) {
        val qHead = kv * LLAMA_GQA + r
        appendMatrixWeights(qOut, s1Out, s2Out, wqQ, wqS1, wqS2, qHead * LLAMA_HC, LLAMA_HC, LLAMA_C)
      }
    }

    appendMatrixWeights(qOut, s1Out, s2Out,
      i8("MHA_WO_Q", LLAMA_C * LLAMA_C),
      i8("MHA_WO_S1", LLAMA_C * (LLAMA_C / G)),
      i8("MHA_WO_S2", LLAMA_C * (LLAMA_C / G)),
      0, LLAMA_C, LLAMA_C)
    val wuQ  = i8("MLP_WU_Q",  LLAMA_CM * LLAMA_C)
    val wuS1 = i8("MLP_WU_S1", LLAMA_CM * (LLAMA_C / G))
    val wuS2 = i8("MLP_WU_S2", LLAMA_CM * (LLAMA_C / G))
    val wgQ  = i8("MLP_WG_Q",  LLAMA_CM * LLAMA_C)
    val wgS1 = i8("MLP_WG_S1", LLAMA_CM * (LLAMA_C / G))
    val wgS2 = i8("MLP_WG_S2", LLAMA_CM * (LLAMA_C / G))
    for (cot <- 0 until LLAMA_CM / G) {
      // PERMUTE consumes the gated MLP up-projection as WU tile, WG tile, repeated
      // per output-channel tile. Keeping DDR in that order avoids a large replay FIFO.
      appendMatrixWeights(qOut, s1Out, s2Out, wuQ, wuS1, wuS2, cot * G, G, LLAMA_C)
      appendMatrixWeights(qOut, s1Out, s2Out, wgQ, wgS1, wgS2, cot * G, G, LLAMA_C)
    }
    appendMatrixWeights(qOut, s1Out, s2Out,
      i8("MLP_WD_Q", LLAMA_C * LLAMA_CM),
      i8("MLP_WD_S1", LLAMA_C * (LLAMA_CM / G)),
      i8("MLP_WD_S2", LLAMA_C * (LLAMA_CM / G)),
      0, LLAMA_C, LLAMA_CM)

    (qOut.toArray, s1Out.toArray, s2Out.toArray)
  }

  def collectLlmWeights(): Array[Byte] = {
    val allLo = Array.ofDim[Byte](LLAMA_L * decoderShape.bytesHalf)
    val allHi = Array.ofDim[Byte](LLAMA_L * decoderShape.bytesHalf)
    for (l <- 0 until LLAMA_L) {
      val (wq, ws1, ws2) = collectLlmRawLayer(l)
      val condensed = composePackedLayer(wq, ws1, ws2, decoderShape)
      val (lo, hi) = splitLoHi(condensed, decoderShape)
      Array.copy(lo, 0, allLo, l * decoderShape.bytesHalf, decoderShape.bytesHalf)
      Array.copy(hi, 0, allHi, l * decoderShape.bytesHalf, decoderShape.bytesHalf)
    }
    allLo ++ allHi
  }

  def collectVitRawLayer(l: Int): (Array[Byte], Array[Byte], Array[Byte]) = {
    val dir = s"$VIT_BIN_PREFIX$l"
    def i8(binName: String, txtName: String, expected: Int): Array[Byte] =
      readInt8MaybeBin(s"$dir/$binName.bin", s"$dir/$txtName.txt", expected)

    val qOut = ArrayBuffer[Byte]()
    val s1Out = ArrayBuffer[Byte]()
    val s2Out = ArrayBuffer[Byte]()

    val qkv = Seq("Q" -> "WQ", "K" -> "WK", "V" -> "WV").map { case (name, tag) =>
      (name,
        i8(s"MHA_${tag}_Q", s"MHA_${tag}_Q", VIT_C * VIT_C),
        i8(s"MHA_${tag}_S1", s"MHA_${tag}_S1", VIT_C * (VIT_C / G)),
        i8(s"MHA_${tag}_S2", s"MHA_${tag}_S2", VIT_C * (VIT_C / G)))
    }
    for (h <- 0 until VIT_C / VIT_HC; (_, q, s1, s2) <- qkv) {
      // ViT PERMUTE 的 QKV 权重消费顺序是 head -> Q/K/V -> TT；
      // DDR compact 权重去掉 TT 重放后仍必须保持 head -> Q/K/V。
      appendMatrixWeights(qOut, s1Out, s2Out, q, s1, s2, h * VIT_HC, VIT_HC, VIT_C)
    }

    appendMatrixWeights(
      qOut, s1Out, s2Out,
      i8("MHA_WO_Q", "MHA_WO_Q", VIT_C * VIT_C),
      i8("MHA_WO_S1", "MHA_WO_S1", VIT_C * (VIT_C / G)),
      i8("MHA_WO_S2", "MHA_WO_S2", VIT_C * (VIT_C / G)),
      0, VIT_C, VIT_C
    )
    appendMatrixWeights(
      qOut, s1Out, s2Out,
      i8("MLP_WFC1_Q", "MLP_W1_Q", VIT_CM * VIT_C),
      i8("MLP_WFC1_S1", "MLP_W1_S1", VIT_CM * (VIT_C / G)),
      i8("MLP_WFC1_S2", "MLP_W1_S2", VIT_CM * (VIT_C / G)),
      0, VIT_CM, VIT_C
    )
    appendMatrixWeights(
      qOut, s1Out, s2Out,
      i8("MLP_WFC2_Q", "MLP_W2_Q", VIT_C * VIT_CM),
      i8("MLP_WFC2_S1", "MLP_W2_S1", VIT_C * (VIT_CM / G)),
      i8("MLP_WFC2_S2", "MLP_W2_S2", VIT_C * (VIT_CM / G)),
      0, VIT_C, VIT_CM
    )

    (qOut.toArray, s1Out.toArray, s2Out.toArray)
  }

  def collectVitWeights(): Array[Byte] = {
    val allLo = Array.ofDim[Byte](VIT_L * vitShape.bytesHalf)
    val allHi = Array.ofDim[Byte](VIT_L * vitShape.bytesHalf)
    for (l <- 0 until VIT_L) {
      val condensedPath = s"${ctrl_cfg.vit_condense_prefix}$l/CONDENSED_WEIGHT.bin"
      val condensed =
        if (new File(condensedPath).isFile) {
          println(s"Using $condensedPath")
          read_int8_file(condensedPath, vitShape.bytes)
        } else {
          println(s"Composing ViT vision_$l from raw binaries/txt")
          val (wq, ws1, ws2) = collectVitRawLayer(l)
          composePackedLayer(wq, ws1, ws2, vitShape)
        }
      val (lo, hi) = splitLoHi(condensed, vitShape)
      Array.copy(lo, 0, allLo, l * vitShape.bytesHalf, vitShape.bytesHalf)
      Array.copy(hi, 0, allHi, l * vitShape.bytesHalf, vitShape.bytesHalf)
    }
    allLo ++ allHi
  }

  def collectLlmLnw(): Array[Int] = {
    val mergedPath = s"$LLM_REF_PATH/RMSNORM_LNW.txt"
    val merged = readTextInts(mergedPath)
    if (merged.length == LLAMA_L * 2 * LLAMA_C) {
      merged
    } else {
      // 兼容只提供 MHA/MLP 两个文件的情况：按 layer 交错成 MHA0, MLP0, MHA1, MLP1...
      val mha = readTextInts(s"$LLM_REF_PATH/MHA_RMSNORM_LNW.txt")
      val mlp = readTextInts(s"$LLM_REF_PATH/MLP_RMSNORM_LNW.txt")
      require(mha.length == LLAMA_L * LLAMA_C, s"Unexpected MHA_RMSNORM_LNW length: ${mha.length}")
      require(mlp.length == LLAMA_L * LLAMA_C, s"Unexpected MLP_RMSNORM_LNW length: ${mlp.length}")
      val out = Array.ofDim[Int](LLAMA_L * 2 * LLAMA_C)
      for (l <- 0 until LLAMA_L) {
        Array.copy(mha, l * LLAMA_C, out, (l * 2 + 0) * LLAMA_C, LLAMA_C)
        Array.copy(mlp, l * LLAMA_C, out, (l * 2 + 1) * LLAMA_C, LLAMA_C)
      }
      out
    }
  }

  def collectClsLnw(): Array[Int] = {
    val cls = readTextInts(s"$LLM_REF_PATH/CLS_RMSNORM_LNW.txt")
    require(cls.length == LLAMA_C, s"Unexpected CLS_RMSNORM_LNW length: ${cls.length}")
    cls
  }

  def collectVitBias(): Array[Int] = {
    val out = ArrayBuffer[Int]()
    for (l <- 0 until VIT_L) {
      if (l == 0 && new File(s"$VIT_REF_PATH/attn_layer0_bq.txt").isFile) {
        out ++= readTextLongs(s"$VIT_REF_PATH/attn_layer0_bq.txt").map(_.toInt)
        out ++= readTextLongs(s"$VIT_REF_PATH/attn_layer0_bk.txt").map(_.toInt)
        out ++= readTextLongs(s"$VIT_REF_PATH/attn_layer0_bv.txt").map(_.toInt)
        out ++= readTextLongs(s"$VIT_REF_PATH/attn_layer0_bo.txt").map(_.toInt)
        out ++= readTextLongs(s"$VIT_REF_PATH/mlp_layer0_b1.txt").map(_.toInt)
        out ++= readTextLongs(s"$VIT_REF_PATH/mlp_layer0_b2.txt").map(_.toInt)
      } else {
        val dir = s"$VIT_BIN_PREFIX$l"
        out ++= readI64MaybeBin(s"$dir/MHA_BQ.bin", s"$dir/MHA_BQ.txt", VIT_C).map(_.toInt)
        out ++= readI64MaybeBin(s"$dir/MHA_BK.bin", s"$dir/MHA_BK.txt", VIT_C).map(_.toInt)
        out ++= readI64MaybeBin(s"$dir/MHA_BV.bin", s"$dir/MHA_BV.txt", VIT_C).map(_.toInt)
        out ++= readI64MaybeBin(s"$dir/MHA_BO.bin", s"$dir/MHA_BO.txt", VIT_C).map(_.toInt)
        out ++= readI64MaybeBin(s"$dir/MLP_B1.bin", s"$dir/MLP_B1.txt", VIT_CM).map(_.toInt)
        out ++= readI64MaybeBin(s"$dir/MLP_B2.bin", s"$dir/MLP_B2.txt", VIT_C).map(_.toInt)
      }
    }
    require(out.length == VIT_L * (3 * VIT_C + VIT_C + VIT_CM + VIT_C), s"Unexpected ViT bias length ${out.length}")
    out.toArray
  }

  def collectVitLnw(): Array[Int] = {
    val merged = readTextInts(s"$VIT_REF_PATH/VIT_LAYERNORM_LNW.txt")
    require(merged.length == VIT_L * 2 * VIT_C, s"Unexpected VIT_LAYERNORM_LNW length: ${merged.length}")
    merged
  }

  def collectVitLnb(): Array[Long] = {
    val merged = readTextLongs(s"$VIT_REF_PATH/VIT_LAYERNORM_LNB.txt")
    require(merged.length == VIT_L * 2 * VIT_C, s"Unexpected VIT_LAYERNORM_LNB length: ${merged.length}")
    merged
  }

  ensureDir(ctrl_cfg.packed_path)

  if (selected("llm")) {
    writeFile(
      s"${ctrl_cfg.packed_path}/all_decoder_w.bin",
      collectLlmWeights()
    )
    writeInt32File(
      s"${ctrl_cfg.packed_path}/all_llm_lnw_i32.bin",
      collectLlmLnw()
    )
  }

  if (selected("cls")) {
    writeFile(
      s"${ctrl_cfg.packed_path}/all_cls_w.bin",
      collectSingleWeight(s"${ctrl_cfg.condense_prefix}$LLAMA_L/CONDENSED_WEIGHT.bin", clsShape)
    )
    writeInt32File(
      s"${ctrl_cfg.packed_path}/cls_lnw_i32.bin",
      collectClsLnw()
    )
  }

  if (selected("vit")) {
    writeFile(
      s"${ctrl_cfg.packed_path}/all_vit_w.bin",
      collectVitWeights()
    )
    writeInt32File(
      s"${ctrl_cfg.packed_path}/all_vit_bias_i32.bin",
      collectVitBias()
    )
    writeInt32File(
      s"${ctrl_cfg.packed_path}/all_vit_lnw_i32.bin",
      collectVitLnw()
    )
    writeInt64File(
      s"${ctrl_cfg.packed_path}/all_vit_lnb_i64.bin",
      collectVitLnb()
    )
  }
}
