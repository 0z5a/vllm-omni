"""Keep validation workers joinable and prohibit process termination signals."""

import os
from multiprocessing.process import BaseProcess

original_kill = os.kill
original_start = BaseProcess.start


def kill(pid: int, signal: int) -> None:
    if signal:
        raise RuntimeError(f"Process signals are prohibited: pid={pid}, signal={signal}")
    original_kill(pid, signal)


def start(process: BaseProcess) -> None:
    process.daemon = False
    original_start(process)


def reject_signal(process: BaseProcess) -> None:
    raise RuntimeError(f"Process termination is prohibited: pid={process.pid}")


os.kill = kill
BaseProcess.start = start
BaseProcess.terminate = reject_signal
BaseProcess.kill = reject_signal

# These changes affect only the private validation process, after timing.
# Join the Orchestrator before interpreter exit rather than abandoning it.
from vllm_omni.engine import omni_engine_base

omni_engine_base.SHUTDOWN_JOIN_TIMEOUT_S = None

if os.environ.get("Q21_TRACE_ONLY") == "1":
    from vllm_omni.profiler.omni_torch_profiler import OmniTorchProfilerWrapper

    # Chrome trace export still runs in _on_trace_ready; no Excel/stack tables.
    OmniTorchProfilerWrapper._on_stop_hook = lambda self: None
