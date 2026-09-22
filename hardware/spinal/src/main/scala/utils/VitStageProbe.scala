package utils

import spinal.core._

import scala.language.postfixOps

case class VitStageProbeStatus() extends Bundle {
  val flags = UInt(32 bits)
  val layerCycles = UInt(32 bits)
  val aFirstW = UInt(32 bits)
  val aLastB = UInt(32 bits)
  val aFirstAr = UInt(32 bits)
  val aLastR = UInt(32 bits)
  val xmFirstW = UInt(32 bits)
  val xmLastB = UInt(32 bits)
  val xmFirstAr = UInt(32 bits)
  val xmLastR = UInt(32 bits)

  val permuteStart = Vec(UInt(32 bits), 4)
  val permuteDone = Vec(UInt(32 bits), 4)
  val qkStart = Vec(UInt(32 bits), 1)
  val qkDone = Vec(UInt(32 bits), 1)
  val softmaxStart = Vec(UInt(32 bits), 1)
  val softmaxDone = Vec(UInt(32 bits), 1)
  val rvStart = Vec(UInt(32 bits), 1)
  val rvDone = Vec(UInt(32 bits), 1)
  val aReorderStart = Vec(UInt(32 bits), 1)
  val aReorderDone = Vec(UInt(32 bits), 1)
  val xmReorderStart = Vec(UInt(32 bits), 1)
  val xmReorderDone = Vec(UInt(32 bits), 1)
  val rmsStart = Vec(UInt(32 bits), 2)
  val rmsDone = Vec(UInt(32 bits), 2)
  val siluStart = Vec(UInt(32 bits), 1)
  val siluDone = Vec(UInt(32 bits), 1)
  val residualStart = Vec(UInt(32 bits), 2)
  val residualDone = Vec(UInt(32 bits), 2)
}

class VitStageProbe extends Component {
  val io = new Bundle {
    val launch = in Bool()
    val allIdle = in Bool()

    val aWFire = in Bool()
    val aBFire = in Bool()
    val aArFire = in Bool()
    val aRFire = in Bool()
    val xmWFire = in Bool()
    val xmBFire = in Bool()
    val xmArFire = in Bool()
    val xmRFire = in Bool()

    val status = out(VitStageProbeStatus())
  }

  noIoPrefix()

  private val cycle = Reg(UInt(32 bits)) init(0)
  private val active = Reg(Bool()) init(False)
  private val seenBusy = Reg(Bool()) init(False)
  private val overflow = Reg(Bool()) init(False)
  private val layerCycles = Reg(UInt(32 bits)) init(0)

  private val aFirstW = Reg(UInt(32 bits)) init(0)
  private val aLastB = Reg(UInt(32 bits)) init(0)
  private val aFirstAr = Reg(UInt(32 bits)) init(0)
  private val aLastR = Reg(UInt(32 bits)) init(0)
  private val aSawWriteData = Reg(Bool()) init(False)
  private val aSawRead = Reg(Bool()) init(False)
  private val xmFirstW = Reg(UInt(32 bits)) init(0)
  private val xmLastB = Reg(UInt(32 bits)) init(0)
  private val xmFirstAr = Reg(UInt(32 bits)) init(0)
  private val xmLastR = Reg(UInt(32 bits)) init(0)
  private val xmSawWriteData = Reg(Bool()) init(False)
  private val xmSawRead = Reg(Bool()) init(False)

  when(io.launch) {
    cycle := 0
    active := True
    seenBusy := False
    overflow := False
    layerCycles := 0
    aFirstW := 0
    aLastB := 0
    aFirstAr := 0
    aLastR := 0
    aSawWriteData := False
    aSawRead := False
    xmFirstW := 0
    xmLastB := 0
    xmFirstAr := 0
    xmLastR := 0
    xmSawWriteData := False
    xmSawRead := False
  } otherwise {
    when(active) {
      cycle := cycle + 1
      when(!io.allIdle) {
        seenBusy := True
      }
      when(seenBusy && io.allIdle) {
        layerCycles := cycle
        active := False
      }
      when(io.aWFire && !aSawWriteData) {
        aFirstW := cycle
        aSawWriteData := True
      }
      when(io.aBFire) {
        aLastB := cycle
      }
      when(io.aArFire && !aSawRead) {
        aFirstAr := cycle
        aSawRead := True
      }
      when(io.aRFire) {
        aLastR := cycle
      }
      when(io.xmWFire && !xmSawWriteData) {
        xmFirstW := cycle
        xmSawWriteData := True
      }
      when(io.xmBFire) {
        xmLastB := cycle
      }
      when(io.xmArFire && !xmSawRead) {
        xmFirstAr := cycle
        xmSawRead := True
      }
      when(io.xmRFire) {
        xmLastR := cycle
      }
    }
  }

  io.status.flags := 0
  io.status.flags(0) := overflow
  io.status.flags(1) := active
  io.status.layerCycles := layerCycles
  io.status.aFirstW := aFirstW
  io.status.aLastB := aLastB
  io.status.aFirstAr := aFirstAr
  io.status.aLastR := aLastR
  io.status.xmFirstW := xmFirstW
  io.status.xmLastB := xmLastB
  io.status.xmFirstAr := xmFirstAr
  io.status.xmLastR := xmLastR
  io.status.permuteStart.foreach(_ := 0)
  io.status.permuteDone.foreach(_ := 0)
  io.status.qkStart.foreach(_ := 0)
  io.status.qkDone.foreach(_ := 0)
  io.status.softmaxStart.foreach(_ := 0)
  io.status.softmaxDone.foreach(_ := 0)
  io.status.rvStart.foreach(_ := 0)
  io.status.rvDone.foreach(_ := 0)
  io.status.aReorderStart.foreach(_ := 0)
  io.status.aReorderDone.foreach(_ := 0)
  io.status.xmReorderStart.foreach(_ := 0)
  io.status.xmReorderDone.foreach(_ := 0)
  io.status.rmsStart.foreach(_ := 0)
  io.status.rmsDone.foreach(_ := 0)
  io.status.siluStart.foreach(_ := 0)
  io.status.siluDone.foreach(_ := 0)
  io.status.residualStart.foreach(_ := 0)
  io.status.residualDone.foreach(_ := 0)
}
