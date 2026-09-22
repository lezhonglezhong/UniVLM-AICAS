package utils

import spinal.core._
import spinal.core.sim._
import spinal.lib._
import spinal.lib.bus.amba4.axilite._

import scala.language.postfixOps

// @formatter:off
class Controller extends Component {
  val io = new Bundle {
    val axilite: AxiLite4 = slave(AxiLite4(addressWidth = ctrl_cfg.axilite_addr_width, dataWidth = ctrl_cfg.axilite_data_width))
    val signals: ManagerSignals = out(ManagerSignals())
    val idle: Bool = in Bool()
    val vitProbe: VitStageProbeStatus = in(VitStageProbeStatus())
  }
  noIoPrefix()
  AxiLite4SpecRenamer(io.axilite)

  // hyperparameters
  val T = 8

  // data registers
  private val busCtrl = new AxiLite4SlaveFactory(io.axilite)
  io.signals.T := False // on write requires default value
  busCtrl.driveAndRead(io.signals.L_BEGIN,              ctrl_cfg.ADDR_L_BEGIN,              0, "Start layer")
  busCtrl.driveAndRead(io.signals.L_CLOSE,              ctrl_cfg.ADDR_L_CLOSE,              0, "Close layer")
  busCtrl.driveAndRead(io.signals.MODE,                 ctrl_cfg.ADDR_MODE,                 0, "Reuse mode")
  busCtrl.driveAndRead(io.signals.PARAM_OP,             ctrl_cfg.ADDR_PARAM_OP,             0, "Param mover op")
  busCtrl.driveAndRead(io.signals.STATE_OP,             ctrl_cfg.ADDR_STATE_OP,             0, "State mover op")
  busCtrl.driveAndRead(io.signals.RUN_MASK,             ctrl_cfg.ADDR_RUN_MASK,             0, "Run mask")
  busCtrl.driveAndRead(io.signals.CACHE_UPDATE_MODE,    ctrl_cfg.ADDR_CACHE_UPDATE_MODE,    0, "Cache update")
  busCtrl.driveAndRead(io.signals.POS,                  ctrl_cfg.ADDR_POS,                  0, "Position"   )
  busCtrl.driveAndRead(io.signals.MEMORY_DECODER_X,     ctrl_cfg.ADDR_MEMORY_DECODER_X,     0, "Memory X"   )
  busCtrl.driveAndRead(io.signals.MEMORY_DECODER_Y,     ctrl_cfg.ADDR_MEMORY_DECODER_Y,     0, "Memory Y"   )
  busCtrl.driveAndRead(io.signals.MEMORY_CLS_Y,         ctrl_cfg.ADDR_MEMORY_CLS_Y,         0, "Memory CLS" )
  busCtrl.driveAndRead(io.signals.MEMORY_DECODER_W_LO,  ctrl_cfg.ADDR_MEMORY_DECODER_W_LO,  0, "Memory W"   )
  busCtrl.driveAndRead(io.signals.MEMORY_DECODER_W_HI,  ctrl_cfg.ADDR_MEMORY_DECODER_W_HI,  0, "Memory W"   )
  busCtrl.driveAndRead(io.signals.MEMORY_CLS_W_LO,      ctrl_cfg.ADDR_MEMORY_CLS_W_LO,      0, "Memory CLS" )
  busCtrl.driveAndRead(io.signals.MEMORY_CLS_W_HI,      ctrl_cfg.ADDR_MEMORY_CLS_W_HI,      0, "Memory CLS" )
  busCtrl.driveAndRead(io.signals.MEMORY_K_CACHE,       ctrl_cfg.ADDR_MEMORY_K_CACHE,       0, "Memory K"   )
  // 复用版额外 DDR 区域：参数、state 与 ViT 权重分开调度，避免沿用 Qwen 单 M_AXI 地址窗口。
  busCtrl.driveAndRead(io.signals.MEMORY_VIT_BIAS,      ctrl_cfg.ADDR_MEMORY_VIT_BIAS,      0, "ViT bias"   )
  busCtrl.driveAndRead(io.signals.MEMORY_LLM_LNW,       ctrl_cfg.ADDR_MEMORY_LLM_LNW,       0, "LLM lnw"    )
  busCtrl.driveAndRead(io.signals.MEMORY_CLS_LNW,       ctrl_cfg.ADDR_MEMORY_CLS_LNW,       0, "CLS lnw"    )
  busCtrl.driveAndRead(io.signals.MEMORY_VIT_LNW,       ctrl_cfg.ADDR_MEMORY_VIT_LNW,       0, "ViT lnw"    )
  busCtrl.driveAndRead(io.signals.MEMORY_VIT_LNB,       ctrl_cfg.ADDR_MEMORY_VIT_LNB,       0, "ViT lnb"    )
  busCtrl.driveAndRead(io.signals.MEMORY_DECODER_STATE, ctrl_cfg.ADDR_MEMORY_DECODER_STATE, 0, "LLM state"  )
  busCtrl.driveAndRead(io.signals.MEMORY_VIT_STATE,     ctrl_cfg.ADDR_MEMORY_VIT_STATE,     0, "ViT state"  )
  busCtrl.driveAndRead(io.signals.MEMORY_VIT_W_LO,      ctrl_cfg.ADDR_MEMORY_VIT_W_LO,      0, "ViT W lo"   )
  busCtrl.driveAndRead(io.signals.MEMORY_VIT_W_HI,      ctrl_cfg.ADDR_MEMORY_VIT_W_HI,      0, "ViT W hi"   )
  busCtrl.driveAndRead(io.signals.MEMORY_VIT_A,         ctrl_cfg.ADDR_MEMORY_VIT_A,         0, "ViT A"      )
  busCtrl.driveAndRead(io.signals.MEMORY_VIT_XM,        ctrl_cfg.ADDR_MEMORY_VIT_XM,        0, "ViT XM"     )
  busCtrl.write       (io.signals.T,                    ctrl_cfg.ADDR_T,                    0, "Trigger"    )
  busCtrl.read        (io.idle,                         ctrl_cfg.ADDR_IDLE,                 0, "Idle"       )

  val vitProbe = RegNext(io.vitProbe)

  private def packedPair(start: UInt, done: UInt): UInt = {
    val packed = UInt(64 bits)
    packed := (done.asBits ## start.asBits).asUInt
    packed
  }

  busCtrl.read(vitProbe.flags.resize(64),      ctrl_cfg.ADDR_VIT_PROBE_FLAGS,       0, "ViT probe flags")
  busCtrl.read(vitProbe.layerCycles.resize(64),ctrl_cfg.ADDR_VIT_PROBE_LAYER,       0, "ViT probe layer cycles")
  busCtrl.read(vitProbe.aFirstW.resize(64),    ctrl_cfg.ADDR_VIT_PROBE_A_FIRST_W,   0, "ViT A first W")
  busCtrl.read(vitProbe.aLastB.resize(64),     ctrl_cfg.ADDR_VIT_PROBE_A_LAST_B,    0, "ViT A last B")
  busCtrl.read(vitProbe.aFirstAr.resize(64),   ctrl_cfg.ADDR_VIT_PROBE_A_FIRST_AR,  0, "ViT A first AR")
  busCtrl.read(vitProbe.aLastR.resize(64),     ctrl_cfg.ADDR_VIT_PROBE_A_LAST_R,    0, "ViT A last R")
  busCtrl.read(vitProbe.xmFirstW.resize(64),   ctrl_cfg.ADDR_VIT_PROBE_XM_FIRST_W,  0, "ViT XM first W")
  busCtrl.read(vitProbe.xmLastB.resize(64),    ctrl_cfg.ADDR_VIT_PROBE_XM_LAST_B,   0, "ViT XM last B")
  busCtrl.read(vitProbe.xmFirstAr.resize(64),  ctrl_cfg.ADDR_VIT_PROBE_XM_FIRST_AR, 0, "ViT XM first AR")
  busCtrl.read(vitProbe.xmLastR.resize(64),    ctrl_cfg.ADDR_VIT_PROBE_XM_LAST_R,   0, "ViT XM last R")

  private val probePairs = Seq.empty[UInt]

  for ((pair, idx) <- probePairs.zipWithIndex) {
    busCtrl.read(pair, ctrl_cfg.ADDR_VIT_PROBE_PAIR_BASE + idx * ctrl_cfg.ADDR_VIT_PROBE_PAIR_STRIDE, 0, "ViT stage pair")
  }

  io.signals.CHUNK := (io.signals.POS >> log2Up(T)).resized
}

object simulate_controller extends App {

  val spinalConfig: SpinalConfig = SpinalConfig(
    defaultConfigForClockDomains = ClockDomainConfig(
      resetKind = ASYNC,
      resetActiveLevel = LOW
    )
  )

  SpinalVerilog(spinalConfig)(new Controller)

  SimConfig
    .withConfig(spinalConfig)
    .withFstWave
    .withWaveDepth(3)
    .compile(new Controller)
    .doSimUntilVoid { dut =>

      val drv = AddressSeparableAxiLite4Driver(dut.io.axilite, dut.clockDomain)
      drv.reset()

      // init
      init_clock(dut.clockDomain, 10)

      drv.write(ctrl_cfg.ADDR_L_BEGIN, 1)
      dut.clockDomain.waitSampling(10)
      drv.write(ctrl_cfg.ADDR_L_CLOSE, 10)
      dut.clockDomain.waitSampling(100)

      drv.write(ctrl_cfg.ADDR_MEMORY_DECODER_X, 0x12345678)
      dut.clockDomain.waitSampling(10)
      drv.write(ctrl_cfg.ADDR_MEMORY_DECODER_W_LO, 0x0EADBEEF)
      dut.clockDomain.waitSampling(10)
      drv.write(ctrl_cfg.ADDR_MEMORY_DECODER_W_HI, 0xCAFEBABE)
      dut.clockDomain.waitSampling(10)
      drv.write(ctrl_cfg.ADDR_MEMORY_DECODER_Y, 0x07654321)
      dut.clockDomain.waitSampling(100)

      drv.write(ctrl_cfg.ADDR_POS, 0x12345678)
      dut.clockDomain.waitSampling(10)

      drv.write(ctrl_cfg.ADDR_T, 1)
      dut.clockDomain.waitSampling(100)

      simSuccess()
    }
}
