# API Server

vLLM-Omni exposes OpenAI-compatible HTTP APIs plus vLLM-Omni extensions for
image, audio, video, and realtime workloads. Use this page to choose an
endpoint; follow the linked reference page for model-specific fields and
response details.

Each server instance hosts one model. An endpoint is usable only when the
loaded model supports that task.

This guide focuses on vLLM-Omni's primary task endpoints. Compatible models
may also expose standard routes inherited from vLLM, such as
`POST /v1/completions`; those routes remain model- and task-dependent.

## Start and Verify the Server

Start a model with the shared serving command:

```bash
vllm serve <model> --omni --port 8000
```

Some models require additional flags or a deployment configuration. Use the
model-specific guide when one is provided.

After the server starts, check its health and served model name:

```bash
export VLLM_OMNI_BASE_URL=http://localhost:8000

curl "$VLLM_OMNI_BASE_URL/health"
curl "$VLLM_OMNI_BASE_URL/v1/models" | jq .
```

Use `http://localhost:8000/v1` as the `base_url` for an OpenAI SDK client. If
the server was started with `--api-key`, include an `Authorization` header with
`Bearer <api-key>` in requests.

## Choose an Endpoint

Prefer a task-specific endpoint when one matches your workload. Use Chat
Completions for conversational or heterogeneous omni pipelines, rather than as
the default wrapper for every generation task.

| Task | Endpoint | Request | Response | Details |
|------|----------|---------|----------|---------|
| Conversation or multimodal understanding/generation | `POST /v1/chat/completions` | JSON | OpenAI-style JSON or SSE | [Chat Completions](chat_completions_api.md) |
| Text-to-speech | `POST /v1/audio/speech` | JSON | Audio bytes or SSE | [Speech](speech_api.md) |
| Sound, music, or ambient audio generation | `POST /v1/audio/generate` | JSON | Audio bytes | [Audio Generation](audio_generate_api.md) |
| Text-to-image | `POST /v1/images/generations` | JSON | Base64 JSON or an image file | [Image Generation](image_generation_api.md) |
| Image editing | `POST /v1/images/edits` | Multipart form | Base64 JSON or SSE | [Image Edit](image_edit_api.md) |
| Video generation (recommended) | `POST /v1/videos` | Multipart form | Asynchronous job JSON | [Videos](videos_api.md) |
| Video generation (blocking) | `POST /v1/videos/sync` | Multipart form | Video bytes | [Videos](videos_api.md#synchronous-response) |
| Live video understanding | `WS /v1/video/chat/stream` | WebSocket events | Streaming text/audio events | [Streaming Video Input](video_stream_api.md) |

`POST /v1/videos/sync` is intended for tests and simple scripts. Use the
asynchronous job API for production requests so clients can poll, download,
and delete generated videos independently.

## Minimal Requests

The examples below assume a compatible model is already running. TTS voices,
media inputs, generation controls, and output capabilities vary by model.

=== "Chat"

    ```bash
    curl "$VLLM_OMNI_BASE_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [{"role": "user", "content": "Describe vLLM-Omni briefly."}],
        "modalities": ["text"]
      }'
    ```

=== "Speech"

    ```bash
    curl "$VLLM_OMNI_BASE_URL/v1/audio/speech" \
      -H "Content-Type: application/json" \
      -d '{
        "input": "Hello from vLLM-Omni.",
        "voice": "vivian"
      }' \
      --output speech.wav
    ```

=== "Audio"

    ```bash
    curl "$VLLM_OMNI_BASE_URL/v1/audio/generate" \
      -H "Content-Type: application/json" \
      -d '{
        "input": "Ocean waves on a quiet beach",
        "audio_length": 5.0
      }' \
      --output audio.wav
    ```

=== "Image"

    ```bash
    curl "$VLLM_OMNI_BASE_URL/v1/images/generations" \
      -H "Content-Type: application/json" \
      -d '{
        "prompt": "A small robot reading beside a window",
        "size": "1024x1024",
        "response_format": "file"
      }' \
      --output image.png
    ```

=== "Image edit"

    ```bash
    curl "$VLLM_OMNI_BASE_URL/v1/images/edits" \
      -F "image=@input.png" \
      -F "prompt=Turn the background into a snowy mountain" \
      | jq -r '.data[0].b64_json' \
      | base64 --decode > edited.png
    ```

=== "Video"

    ```bash
    curl "$VLLM_OMNI_BASE_URL/v1/videos" \
      -F "prompt=A cinematic tracking shot of a mountain lake at sunrise" \
      -F "size=1280x720" \
      -F "seconds=5"
    ```

    The response contains a video job `id`. Poll and download it with:

    ```bash
    curl "$VLLM_OMNI_BASE_URL/v1/videos/<video_id>"
    curl -L "$VLLM_OMNI_BASE_URL/v1/videos/<video_id>/content" --output video.mp4
    ```

## Related Endpoints

| Purpose | Endpoints | Reference |
|---------|-----------|-----------|
| Discovery and readiness | `GET /health`, `GET /v1/models` | This page |
| Batched conversations | `POST /v1/chat/completions/batch` | [Batch requests](chat_completions_api.md#batch-requests) |
| Batched speech | `POST /v1/audio/speech/batch` | [Batch speech generation](speech_api.md#batch-speech-generation) |
| TTS voice management | `GET/POST /v1/audio/voices`, `DELETE /v1/audio/voices/{name}` | [Voices](speech_api.md#voices-endpoint) |
| Streaming text-to-speech | `WS /v1/audio/speech/stream` | [Streaming speech](speech_api.md#streaming-text-input-websocket) |
| Video job lifecycle | `GET /v1/videos`, `GET/DELETE /v1/videos/{video_id}`, `GET /v1/videos/{video_id}/content` | [Video endpoints](videos_api.md#endpoints) |
| Release and restore stage memory | `POST /v1/omni/sleep`, `POST /v1/omni/wakeup` | [Sleep Mode](../features/sleep_mode.md) |

??? info "Specialized and experimental WebSocket endpoints"

    These routes are enabled only for specific pipelines. Start with the
    associated example instead of treating them as general-purpose APIs.

    | Endpoint | Use case | Example |
    |----------|----------|---------|
    | `WS /v1/realtime` | OpenAI-style realtime audio interaction | [Qwen3-Omni realtime client](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/qwen3_omni) |
    | `WS /v1/realtime/video` | Stream generated video chunks | [Streaming video generation](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/streaming_video_generation) |
    | `WS /v1/duplex` or `WS /v1/realtime?duplex=1` | Experimental full-duplex speech interaction | [Full-duplex examples](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/minicpmo) |
    | `WS /v1/realtime/robot/openpi` | OpenPI-compatible robot policy inference | [DreamZero](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/dreamzero) |

## Compatibility Notes

- A registered path does not mean every model supports it. Unsupported tasks
  return an OpenAI-style error or a WebSocket `error` event.
- Use JSON for Chat, Speech, Audio Generation, and Image Generation. Use
  `multipart/form-data` when uploading images, videos, or other files.
- Generation can take minutes for large diffusion models. Configure client
  timeouts accordingly; prefer asynchronous video jobs over long blocking
  requests.
- Model-specific parameters are vLLM-Omni extensions. Check the endpoint
  reference before assuming an OpenAI SDK exposes them directly.
