"""Use normal shutdown messages and natural joins for validation-owned workers."""

from multiprocessing.process import BaseProcess

from vllm_omni.diffusion import stage_diffusion_proc
from vllm_omni.diffusion.data import SHUTDOWN_MESSAGE
from vllm_omni.diffusion.executor.multiproc_executor import _ExecutorShutdownCleaner


def install():
    original_start = BaseProcess.start

    def start(process):
        # Python's exit handler terminates daemon children. Keep our children
        # joinable even when initialization raises before executor setup.
        process.daemon = False
        return original_start(process)

    def cleanup(cleaner):
        if cleaner.broadcast_mq is not None:
            for _ in range(cleaner.num_workers):
                cleaner.broadcast_mq.enqueue(SHUTDOWN_MESSAGE, timeout=1.0)
            cleaner.broadcast_mq = None
        for process in cleaner.processes or []:
            process.join()
        cleaner.processes = []

    def join(processes, timeout=None):
        for process in processes:
            process.join()

    def reject_signal(process):
        raise RuntimeError(f"Validation cannot signal worker {process.pid}; wait for natural exit")

    BaseProcess.start = start
    BaseProcess.terminate = reject_signal
    BaseProcess.kill = reject_signal
    _ExecutorShutdownCleaner._cleanup = cleanup
    stage_diffusion_proc.shutdown = join
