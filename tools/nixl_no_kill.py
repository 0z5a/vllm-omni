# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validation-only natural shutdown for jobs whose processes must never be signalled."""

from __future__ import annotations

import os
import uuid
from concurrent.futures import Future
from multiprocessing.process import BaseProcess

import zmq
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState


def join_only(processes: list[BaseProcess], timeout: float | None = None) -> None:
    for process in processes:
        process.join(60 if timeout is None else max(timeout, 10))
        if process.is_alive():
            print(f"Retained process {process.name} PID={process.pid} for inspection", flush=True)


def request_natural_shutdown(core: EngineCoreProc) -> None:
    core.shutdown_state = EngineShutdownState.REQUESTED


def retain_process(process: BaseProcess) -> None:
    print(f"Retained process {process.name} PID={process.pid}", flush=True)


def no_parent_signal(signum: int) -> None:
    return


def install() -> None:
    import vllm.v1.engine.utils as engine_utils
    import vllm.v1.utils as process_utils

    # Both normal cleanup and constructor-failure cleanup must retain live jobs.
    process_utils.shutdown = join_only
    engine_utils.shutdown = join_only
    real_kill = os.kill

    def retain_signal(pid: int, sig: int) -> None:
        if sig == 0:
            real_kill(pid, sig)
        else:
            print(f"Suppressed process signal PID={pid} signal={sig}; process retained", flush=True)

    def retain_group_signal(pgid: int, sig: int) -> None:
        print(f"Suppressed process-group signal PGID={pgid} signal={sig}; processes retained", flush=True)

    os.kill = retain_signal
    os.killpg = retain_group_signal
    BaseProcess.terminate = retain_process
    BaseProcess.kill = retain_process
    EngineCoreProc.nixl_validation_shutdown = request_natural_shutdown  # type: ignore[attr-defined]

    from vllm_omni.diffusion import stage_diffusion_proc
    from vllm_omni.engine import stage_engine_core_proc, stage_engine_core_proc_manager, stage_init_utils
    from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

    stage_diffusion_proc.shutdown = join_only
    stage_engine_core_proc_manager.shutdown = join_only
    stage_init_utils.set_death_signal = no_parent_signal
    stage_engine_core_proc.set_death_signal = no_parent_signal
    stage_diffusion_proc.set_death_signal = no_parent_signal

    original_shutdown = StageEngineCoreClient.shutdown

    def natural_client_shutdown(client: StageEngineCoreClient, timeout: float | None = None) -> None:
        if not client._finalizer.alive or client.resources.engine_dead:
            original_shutdown(client, timeout=timeout)
            return
        manager = client.resources.engine_manager
        if manager is not None:
            manager.manager_stopped.set()
        call_id = uuid.uuid4().int >> 64
        client.utility_results[call_id] = Future()
        request = (client.client_index, call_id, "nixl_validation_shutdown", ())
        socket = zmq.Socket.shadow(client.input_socket)
        socket.send_multipart(
            (client.core_engine, EngineCoreRequestType.UTILITY.value, *client.encoder.encode(request))
        )
        original_shutdown(client, timeout=timeout)

    StageEngineCoreClient.shutdown = natural_client_shutdown  # type: ignore[method-assign]
