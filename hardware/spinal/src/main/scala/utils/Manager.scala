package utils

import spinal.core._
import spinal.lib._
import spinal.lib.fsm.{EntryPoint, State, StateMachine}

import scala.language.postfixOps

// @formatter:off
case class ApChain() extends Bundle with IMasterSlave {
  // Xilinx ap chain control signals
  val ap_start:     Bool = Bool()
  val ap_continue:  Bool = Bool()
  val ap_idle:      Bool = Bool()
  val ap_ready:     Bool = Bool()
  val ap_done:      Bool = Bool()

  override def asMaster(): Unit = {
    out(ap_start)
    out(ap_continue)
    in(ap_idle)
    in(ap_ready)
    in(ap_done)
  }

  override def asSlave(): Unit = {
    in(ap_start)
    in(ap_continue)
    out(ap_idle)
    out(ap_ready)
    out(ap_done)
  }
}

object ApChainRenamer {
  def apply(that: ApChain): Unit = {
    def doIt(): Unit = {
      that.flatten.foreach(bt =>
        bt.setName(bt.getName().replace("ap_ctrl_", ""))
      )
    }

    if (Component.current == that.component)
      that.component.addPrePopTask(() => {
        doIt()
      })
    else
      doIt()
  }
}

case class DaisyChain[T <: Data](gen: T) extends Bundle {
  // daisy chain IO
  val I: T = in(cloneOf(gen))
  val O: T = out(cloneOf(gen))
}

object ctrl_cfg {
  val REG_BITS: Int = 32 // how many layers to process

  val axilite_addr_width = 16
  val axilite_data_width = 64 // indexing more than 4GB

  // register space for controller
  val ADDR_L_BEGIN              = 0x0000
  val ADDR_L_CLOSE              = 0x0010
  val ADDR_MEMORY_DECODER_X     = 0x0020
  val ADDR_MEMORY_DECODER_Y     = 0x0030
  val ADDR_MEMORY_CLS_Y         = 0x0040
  val ADDR_MEMORY_DECODER_W_LO  = 0x0050
  val ADDR_MEMORY_DECODER_W_HI  = 0x0060
  val ADDR_MEMORY_CLS_W_LO      = 0x0070
  val ADDR_MEMORY_CLS_W_HI      = 0x0080
  val ADDR_MEMORY_K_CACHE       = 0x0090
  val ADDR_POS                  = 0x00A0
  val ADDR_T                    = 0x00B0
  val ADDR_IDLE                 = 0x00C0
  // 复用版新增控制寄存器：mode 在一次 IP 调用期间保持稳定，op 选择 mover 子功能。
  val ADDR_MODE                 = 0x00D0
  val ADDR_PARAM_OP             = 0x00E0
  val ADDR_STATE_OP             = 0x00F0

  // 复用版新增 DDR 区域地址。旧 Qwen 的 X/Y 地址保留但后续由 STATE_AXI state 地址替代。
  val ADDR_MEMORY_VIT_BIAS      = 0x0100
  val ADDR_MEMORY_LLM_LNW       = 0x0110
  val ADDR_MEMORY_CLS_LNW       = 0x0120
  val ADDR_MEMORY_VIT_LNW       = 0x0130
  val ADDR_MEMORY_VIT_LNB       = 0x0140
  val ADDR_MEMORY_DECODER_STATE = 0x0150
  val ADDR_MEMORY_VIT_STATE     = 0x0160
  val ADDR_MEMORY_VIT_W_LO      = 0x0170
  val ADDR_MEMORY_VIT_W_HI      = 0x0180
  // RUN_MASK 为 0 时保持旧行为：一次触发启动整条链。非 0 时按 bit 精确启动指定 IP，
  // 方便 PYNQ 分别调度参数、权重、state、KV cache 和复用计算模块。
  val ADDR_RUN_MASK             = 0x0190
  val ADDR_MEMORY_VIT_A         = 0x01A0
  val ADDR_MEMORY_VIT_XM        = 0x01B0
  val ADDR_CACHE_UPDATE_MODE    = 0x01C0
  val ADDR_VIT_PROBE_FLAGS      = 0x0200
  val ADDR_VIT_PROBE_LAYER      = 0x0210
  val ADDR_VIT_PROBE_A_FIRST_W  = 0x0220
  val ADDR_VIT_PROBE_A_LAST_B   = 0x0230
  val ADDR_VIT_PROBE_A_FIRST_AR = 0x0240
  val ADDR_VIT_PROBE_A_LAST_R   = 0x0250
  val ADDR_VIT_PROBE_XM_FIRST_W = 0x0260
  val ADDR_VIT_PROBE_XM_LAST_B  = 0x0270
  val ADDR_VIT_PROBE_XM_FIRST_AR= 0x0280
  val ADDR_VIT_PROBE_XM_LAST_R  = 0x0290
  val ADDR_VIT_PROBE_PAIR_BASE  = 0x0300
  val ADDR_VIT_PROBE_PAIR_STRIDE= 0x0010
  val VIT_PROBE_PAIR_COUNT      = 0

  val IP_M_AXI          = 0
  val IP_WEIGHT_AXI     = 1
  val IP_KV_CACHE       = 2
  val IP_STATE_AXI      = 3
  val IP_MUX            = 4
  val IP_PERMUTE        = 5
  val IP_DEMUX          = 6
  val IP_ROPE_QK        = 7
  val IP_QK_GEMM        = 8
  val IP_SOFTMAX        = 9
  val IP_RV_GEMM        = 10
  val IP_SILU_GELU      = 11
  val IP_RESIDUAL       = 12
  val IP_RMS_LAYERNORM  = 13
  val IP_A_REORDER      = 14
  val IP_XM_REORDER     = 15

  // Scala 仿真和权重打包默认从 SPINAL 工作目录访问仓库内 Weights；可用环境变量覆盖。
  val weights_root = sys.env.getOrElse("VLM_WEIGHTS_ROOT", "../Weights")
  val binaries_prefix = sys.env.getOrElse("VLM_LLM_BINARIES_PREFIX", s"$weights_root/LLM/binaries/decoder_")
  val condense_prefix = sys.env.getOrElse("VLM_LLM_CONDENSE_PREFIX", s"$weights_root/LLM/condense/decoder_")
  val vit_condense_prefix = sys.env.getOrElse("VLM_VIT_CONDENSE_PREFIX", s"$weights_root/ViT/condense/vision_")
  val condense_path = sys.env.getOrElse("VLM_LLM_CONDENSE_PATH", s"$weights_root/LLM/condense")
  val vit_condense_path = sys.env.getOrElse("VLM_VIT_CONDENSE_PATH", s"$weights_root/ViT/condense")
  val packed_path = sys.env.getOrElse("VLM_PACKED_WEIGHT_PATH", s"$weights_root/packed")
}

case class ManagerSignals() extends Bundle {
  val L_BEGIN:              UInt = UInt(ctrl_cfg.REG_BITS bits)     // start layer
  val L_CLOSE:              UInt = UInt(ctrl_cfg.REG_BITS bits)     // close layer, exclusive
  val MODE:                 UInt = UInt(1 bits)                     // 0=LLM, 1=ViT
  val PARAM_OP:             UInt = UInt(ctrl_cfg.REG_BITS bits)     // M_AXI 参数搬运 op
  val STATE_OP:             UInt = UInt(ctrl_cfg.REG_BITS bits)     // STATE_AXI state/CLS op
  val RUN_MASK:             UInt = UInt(ctrl_cfg.REG_BITS bits)     // bit mask，选择本次触发的 Spinal/HLS IP
  val CACHE_UPDATE_MODE:    UInt = UInt(1 bits)                     // 0=prefill/full tile, 1=decode/current token

  val MEMORY_DECODER_X:     UInt = UInt(64 bits)                    // memory address
  val MEMORY_DECODER_Y:     UInt = UInt(64 bits)                    // memory address
  val MEMORY_CLS_Y:         UInt = UInt(64 bits)                    // memory address
  val MEMORY_DECODER_W_LO:  UInt = UInt(64 bits)                    // memory address
  val MEMORY_DECODER_W_HI:  UInt = UInt(64 bits)                    // memory address
  val MEMORY_CLS_W_LO:      UInt = UInt(64 bits)                    // memory address
  val MEMORY_CLS_W_HI:      UInt = UInt(64 bits)                    // memory address
  val MEMORY_K_CACHE:       UInt = UInt(64 bits)                    // memory address
  val MEMORY_VIT_BIAS:      UInt = UInt(64 bits)                    // ViT DEMUX bias DDR address
  val MEMORY_LLM_LNW:       UInt = UInt(64 bits)                    // LLM RMSNorm gamma DDR address
  val MEMORY_CLS_LNW:       UInt = UInt(64 bits)                    // LLM final RMSNorm gamma DDR address
  val MEMORY_VIT_LNW:       UInt = UInt(64 bits)                    // ViT LayerNorm gamma DDR address
  val MEMORY_VIT_LNB:       UInt = UInt(64 bits)                    // ViT LayerNorm beta DDR address
  val MEMORY_DECODER_STATE: UInt = UInt(64 bits)                    // LLM memory-backed residual state
  val MEMORY_VIT_STATE:     UInt = UInt(64 bits)                    // ViT memory-backed residual state
  val MEMORY_VIT_W_LO:      UInt = UInt(64 bits)                    // ViT GEMM compact weight low half
  val MEMORY_VIT_W_HI:      UInt = UInt(64 bits)                    // ViT GEMM compact weight high half
  val MEMORY_VIT_A:         UInt = UInt(64 bits)                    // ViT RV A reorder scratch memory
  val MEMORY_VIT_XM:        UInt = UInt(64 bits)                    // ViT GELU/XM reorder scratch memory

  val POS:                  UInt = UInt(ctrl_cfg.REG_BITS bits)     // current pos id
  val CHUNK:                UInt = UInt(ctrl_cfg.REG_BITS bits)     // current chunk id
  val T: Bool = Bool()                                            // Trigger
}

class Manager(enableBit: Int = -1) extends Component {
  val io = new Bundle {
    val signals: DaisyChain[ManagerSignals] = DaisyChain(ManagerSignals())
    val ap_ctrl: ApChain = master(ApChain())
  }

  noIoPrefix()

  io.signals.O := RegNext(io.signals.I) // pass all signals

  io.ap_ctrl.ap_continue  := True  // always continue
  io.ap_ctrl.ap_start     := False // default value

  // RUN_MASK=0 时向后兼容旧单触发链；RUN_MASK 非 0 时只有对应 bit 的 IP 启动。
  val runAll = io.signals.I.RUN_MASK === 0
  val selected = if (enableBit < 0) True else (runAll || io.signals.I.RUN_MASK(enableBit))

  val fsm: StateMachine = new StateMachine {
    val s_idle = new State with EntryPoint
    val s_work = new State

    s_idle.whenIsActive {
      when(io.signals.I.T && selected) {
        goto(s_work)
      }
    } // end of s_idle

    s_work.whenIsActive {
      io.ap_ctrl.ap_start := True
      when(io.ap_ctrl.ap_ready) { // handshake with ap_ready
          goto(s_idle)
      } // else, wait for ap_ready
    } // end of s_work

  } // end of fsm
}

object genManager extends App {
  SpinalConfig(
    defaultConfigForClockDomains = ClockDomainConfig(
      resetKind = ASYNC,
      resetActiveLevel = LOW
    ),
    mode = Verilog
  ).generate(new Manager)
}
