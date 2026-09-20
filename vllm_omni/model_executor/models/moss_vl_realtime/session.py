# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-local session state for one MOSS-VL-Realtime duplex conversation.

This is the piece T09 calls the MOSS session: it owns the token stream, the paced
position cursor, frame publication, the response state and the session budgets.
It deliberately owns *no* loop and *no* registry — the engine drives stepping, so
there is exactly one scheduler and one lifecycle in the process.

Frame encoding is injected as a callable, which keeps this module free of
Transformers: the engine (or a capture harness) passes whatever encoder the
processor stack provides.

Nothing here is wired into the duplex engine yet; the module states and tests the
contract the engine adapter has to satisfy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from .modeling import MossVLNativeModel, SessionLimits

#: The reference session's default system prompt; the silence behaviour is
#: conditioned on it, so a session that omits it is not behaviour-equivalent.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in real-time video analysis. "
    "The video streams to you frame by frame. At every frame, you decide independently "
    "whether to respond or stay silent — output `<|silence|>` when nothing relevant has happened, "
    "and respond when the visual content warrants it."
)

#: ``frame_encoder(path_or_frame, media_timestamp_ms) -> {"pixel_values", "grid_thw", "segment_ids"}``
FrameEncoder = Callable[[Any, int], dict[str, torch.Tensor]]
Detokenize = Callable[[list[int]], str]


@dataclass
class SessionEvent:
    kind: str
    wall: float
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionStats:
    """Why a step decided what it decided.

    The engine shows this: a silence that beat the runner-up by a hair is a
    different operational fact from a silence that won outright.
    """

    token_id: int
    max: float
    top5: tuple[int, ...]

    @classmethod
    def of(cls, logits: torch.Tensor) -> DecisionStats:
        row = logits[0, -1].float()
        return cls(token_id=int(row.argmax()), max=float(row.max()), top5=tuple(int(v) for v in row.topk(5).indices))


@dataclass
class AppendResult:
    """What the caller learns from one append.

    ``published_vision_length`` moves only after the frames are in the cache;
    ``accepted_frames`` counts what the transport was given, which is not the same
    promise.
    """

    accepted_frames: int
    published_vision_length: int
    next_position: int
    segment_tokens: int
    pending_frames: int = 0


class MossVLNativeSession:
    """Caller-driven session over the native executor."""

    def __init__(
        self,
        model: MossVLNativeModel,
        *,
        frame_encoder: FrameEncoder,
        detokenize: Detokenize,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        limits: SessionLimits | None = None,
        max_new_tokens_per_turn: int = 32,
    ) -> None:
        self.model = model
        self.frame_encoder = frame_encoder
        self.detokenize = detokenize
        self.system_prompt = system_prompt
        self.max_new_tokens_per_turn = max_new_tokens_per_turn
        if limits is not None:
            model.limits = limits
        self._logits: torch.Tensor | None = None
        #: The decision made on the most recently consumed input, i.e. what the
        #: session answered the segment it was just given.
        self.segment_decision: DecisionStats | None = None
        #: The decision that ended (or bounded) the most recent turn.
        self.last_decision: DecisionStats | None = None
        self._pending_silence: list[int] = []
        self._output: list[str] = []
        self._events: list[SessionEvent] = []
        self._closed = False
        self._frames_accepted = 0
        self._clock = 0.0

    # ---- lifecycle --------------------------------------------------------
    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def events(self) -> list[SessionEvent]:
        return list(self._events)

    def _record(self, kind: str, **fields: Any) -> None:
        self._clock += 0.001
        self._events.append(SessionEvent(kind=kind, wall=self._clock, fields=fields))

    def start(self, prompt_ids: torch.Tensor) -> None:
        """Prefill the system+user turn. The caller owns prompt rendering."""
        self.model.reset()
        with torch.no_grad():
            self._logits = self.model(prompt_ids, self.model.compute_text_position_ids(prompt_ids), offset=0)
        self.segment_decision = DecisionStats.of(self._logits)
        self._record("session_open", prompt_tokens=int(prompt_ids.shape[1]), next_position=self.model.next_position)

    # ---- input ------------------------------------------------------------
    def append(
        self,
        *,
        frames: list[tuple[Any, int]] | None = None,
        prompts: list[str] | None = None,
        prompt_ids_for: Callable[[str], list[int]] | None = None,
    ) -> AppendResult:
        """Publish frames and/or a new user turn, then run one bounded turn.

        ``prompt_ids_for`` renders a user turn (the checkpoint chat template); it is
        required only when ``prompts`` is non-empty.
        """
        if self.closed:
            raise RuntimeError("session is closed")
        if self._logits is None:
            raise RuntimeError("start() must run before append()")

        frames = frames or []
        prompts = prompts or []
        if not frames and not prompts:
            raise ValueError("an append needs at least one frame or prompt")
        if prompts and prompt_ids_for is None:
            raise ValueError("rendering a prompt needs prompt_ids_for")

        ids: list[int] = list(self._pending_silence)
        for prompt in prompts:
            ids.extend(prompt_ids_for(prompt))
        ids.append(self.model.config.silence_token_id)

        visions: list[dict[str, torch.Tensor]] = []
        for frame, media_timestamp_ms in frames:
            encoded = self.frame_encoder(frame, media_timestamp_ms)
            vision = {key: value.to(self._device) for key, value in encoded.items()}
            ids.extend(vision["segment_ids"][0].tolist())
            visions.append(vision)

        # Check the whole append before consuming or publishing any of it: a refusal
        # halfway through would either drop the carried silence marker or leave frames
        # in the cache that the token stream never referred to.
        self.model.limits.check_text(self.model.text_cache.length(0) + len(ids))
        if visions:
            grid_thw = torch.cat([vision["grid_thw"] for vision in visions], dim=0)
            self.model.limits.check_frames(len(visions))
            self.model.limits.check_vision(self.model.vision_length + self.model.vision_tokens_for(grid_thw))
        self._pending_silence.clear()

        segment = torch.tensor([ids], dtype=torch.long, device=self._device)
        if visions:
            text_positions, _, next_position = self.model.paced_segment_positions(
                segment, grid_thw, self.model.next_position
            )
            slots = (segment[0] == self.model.config.image_token_id).nonzero().flatten().tolist()
            merge = self.model.config.vision.spatial_merge_size
            for slot, vision in zip(slots, visions):
                grid_h = int(vision["grid_thw"][0, 1].item())
                grid_w = int(vision["grid_thw"][0, 2].item())
                max_hw = max(grid_h // merge, grid_w // merge)
                self.model.append_frames(
                    vision["pixel_values"], vision["grid_thw"], int(text_positions[0, 0, slot].item()) - max_hw
                )
                self._frames_accepted += 1
            self.model.next_position = next_position
        else:
            # A prompt-only segment consumes one position per token, exactly like a
            # generated token does.
            text_positions = self._text_positions(segment.shape[1], self.model.next_position)
            self.model.next_position += segment.shape[1]

        with torch.no_grad():
            self._logits = self.model(
                segment,
                text_positions,
                offset=self.model.text_cache.length(0),
                cross_attention_mask=self._visibility(segment, self._frames_accepted),
            )
        self.segment_decision = DecisionStats.of(self._logits)
        self._record(
            "append",
            frames=len(frames),
            prompts=len(prompts),
            segment_tokens=int(segment.shape[1]),
            vision_length=self.model.vision_length,
            next_position=self.model.next_position,
            decision=self.segment_decision.token_id,
        )
        return AppendResult(
            accepted_frames=len(frames),
            published_vision_length=self.model.vision_length,
            next_position=self.model.next_position,
            segment_tokens=int(segment.shape[1]),
        )

    # ---- output -----------------------------------------------------------
    def step(self, budget: int | None = None) -> tuple[str, bool]:
        """Run one bounded turn. Returns the text produced and whether it fell silent.

        A silence decision is not fed here: the reference carries the
        ``<|silence|>`` marker into the segment that follows new input, and feeding it
        twice would desynchronise the stream.
        """
        if self._logits is None:
            raise RuntimeError("start() must run before step()")
        budget = self.max_new_tokens_per_turn if budget is None else budget
        pieces: list[str] = []
        for _ in range(budget):
            self.last_decision = DecisionStats.of(self._logits)
            token_id = self.last_decision.token_id
            if token_id == self.model.config.silence_token_id:
                self._pending_silence.append(token_id)
                self._record("silence", text="".join(pieces), max=round(self.last_decision.max, 4))
                return "".join(pieces), True
            position = self.model.paced_position_ids(advance=1)
            token = torch.tensor([[token_id]], dtype=torch.long, device=self._device)
            with torch.no_grad():
                self._logits = self.model(token, position, offset=self.model.text_cache.length(0))
            pieces.append(self.detokenize([token_id]))
        text = "".join(pieces)
        self._output.append(text)
        self._record("output", text=text)
        return text, False

    def poll_output(self) -> str | None:
        return self._output.pop(0) if self._output else None

    def close(self) -> dict[str, Any]:
        """Release session state and report what the session did."""
        report = {
            "frames_accepted": self._frames_accepted,
            "silences": sum(1 for event in self._events if event.kind == "silence"),
            "events": len(self._events),
            "vision_length_at_close": self.model.vision_length,
            "text_tokens_at_close": self.model.text_cache.length(0),
        }
        self.model.reset()
        self._logits = None
        self._pending_silence.clear()
        self.segment_decision = None
        self.last_decision = None
        self._closed = True
        self._record("session_closed", **report)
        return report

    # ---- internals --------------------------------------------------------
    @property
    def _device(self) -> torch.device:
        return self.model.model.separator_token.device

    def _text_positions(self, length: int, start: int) -> torch.Tensor:
        positions = torch.arange(start, start + length, dtype=torch.long, device=self._device)
        return positions.view(1, 1, length).expand(3, 1, length)

    def _visibility(self, segment: torch.Tensor, total_frames: int) -> torch.Tensor | None:
        """Frame-level visibility, rebuilt from the segment's own tokens.

        ``visible = cum_image_tokens > frame_index``, with the frame index spanning
        every published frame: a row sees exactly the frames whose markers it has
        already passed, so the frame it is consuming is not yet visible to itself.
        The reference rebuilds this per step from that step's tokens, so a segment
        that carries no frame marker leaves every frame invisible rather than
        dropping the mask — the masked rows then take the same path the reference
        gives them.
        """
        if total_frames == 0:
            return None
        seen = (segment[0] == self.model.config.image_token_id).cumsum(0)
        frame_indices = torch.arange(total_frames, device=segment.device)
        visible = seen.unsqueeze(-1) > frame_indices.unsqueeze(0)
        return (~visible).unsqueeze(0).unsqueeze(0)
