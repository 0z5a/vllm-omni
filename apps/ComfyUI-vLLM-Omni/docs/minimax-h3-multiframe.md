# MiniMax H3 Multiframe Reference (WF-04 development draft)

This layout depends on [timeline guide conditioning #8460](https://github.com/vllm-project/vllm-omni/pull/8460),
at `755d73bdbc64b45b6e1327d7636d9ab330ca3324`, for
[RFC #7380, WF-04](https://github.com/vllm-project/vllm-omni/issues/7380).
The model and offline API support ordered timeline guides at that revision.
Remote guide nodes and `/v1/videos` guide serialization remain to be integrated.

The [draft layout](drafts/MiniMax_H3_Multiframe_Reference.draft.json) keeps
Generate Video and Save Video muted. Its three guide notes describe the intended
connections and send no media. The pinned client accepts the default image-only
semantic reference; its ordered reference serializer is already available.

## Four images, three timeline guides

The source is the [official multiframe template](https://github.com/Comfy-Org/workflow_templates/blob/aaac56dd5cc5497533d92cbe50edc35ea660e587/templates/video_minimax_h3_multiframe_reference.json).

| Image | Role | Time | Pixel-frame index |
| --- | --- | --- | --- |
| `h3_frame_ref_1.png` | Reference `<Picture 1>` | Opening | No guide |
| `h3_frame_ref_2.png` | First timeline guide | 1.5 s | 36 |
| `h3_frame_ref_3.png` | Second timeline guide | 3.0 s | 72 |
| `h3_frame_ref_4.png` | Third timeline guide | 5.0 s | 120 |

At 24 FPS, `duration=5.167` requests 124 frames (`17 * 7 + 5`). The duration
widget uses seconds. Still-image guides each occupy one frame; the `17k+5`
clip rule applies to video guides. Negative guide indices count back from the
aligned output end.

Keep guides in order 36 -> 72 -> 120 and separate from semantic references.
They do not enter the Qwen reference presentation or receive `<Picture N>` labels.
The prompt uses `<Picture 1>` only. An opening-composition instruction does not
guarantee exact first-frame preservation.

The upstream offline input for this topology is:

```python
prompt = {
    "prompt": "Use <Picture 1> for the subject identity and opening composition.",
    "multi_modal_data": {
        "image": "h3_frame_ref_1.png",
        "timeline_guides": [
            {"frame_index": 36, "image": "h3_frame_ref_2.png"},
            {"frame_index": 72, "image": "h3_frame_ref_3.png"},
            {"frame_index": 120, "image": "h3_frame_ref_4.png"},
        ],
    },
}
```

This is the model/offline contract, not a ComfyUI API-format prompt or an HTTP
request body. The pinned Generate Video node and video request schema have no
timeline-guide input yet.

## Inputs and server settings

Place the four images in ComfyUI's `input/` directory. Loader widgets use portable
filenames. Pinned official inputs:

- [Image 1](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/aaac56dd5cc5497533d92cbe50edc35ea660e587/input/h3_frame_ref_1.png)
- [Image 2](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/aaac56dd5cc5497533d92cbe50edc35ea660e587/input/h3_frame_ref_2.png)
- [Image 3](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/aaac56dd5cc5497533d92cbe50edc35ea660e587/input/h3_frame_ref_3.png)
- [Image 4](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/aaac56dd5cc5497533d92cbe50edc35ea660e587/input/h3_frame_ref_4.png)

Use base server-format H3 Ref2VA weights, a 1344x768 canvas, explicit
`aspect_ratio=16:9`, 24 FPS and 124 frames. The H3 Params node exposes flow
shifts; verify the named ratio in the integrated request rather than deriving
it from the rounded canvas dimensions.

Use 50 sigma points, `guidance_scale=1`, `true_cfg_scale=1`, `flow_shift=12`,
`audio_flow_shift=3`, seed 738004 and cache-free `quality=lossless`. These are
target settings, not a measured result for this layout.

Timeline guides in #8460 reject cache acceleration, FastH3, active LoRA/Turbo
and distilled schedules. Keep the LoRA and FastH3 inputs disconnected.
Continuation, latent editing/refine/upscale, `audio_mode=lock_source` and a
separate encoder stage are also unsupported with guides at the pinned revision.

## Integration and validation

`extra.wf04` records the dependency revision, image roles and target output.
Its offline field list describes the model contract; the remote guide node and
wire fields remain null until their implementation is available.

1. Add GUIDE-02 nodes and HTTP guide transport over the pinned model contract.
   Preserve image-only semantic references and guide order.
2. Replace the three guide notes with those nodes and connect their final output
   to Generate Video. Test request serialization, negative indices, still images,
   clip lengths, audio and out-of-range rejection.
3. Import the final graph in real ComfyUI and export its API-format prompt.
   Run it against the pinned server. Record source/model revisions, media hashes,
   GPU UUIDs and the exact command.
4. Save through ComfyUI Save Video and check 124 frames, 24 FPS, 1344x768,
   nonempty stereo audio and duration agreement within one video frame. Inspect
   guide placement and generated-media quality.
5. Move the validated graph into `example_workflows/` when real execution passes.

CPU JSON/AST checks validate links, schemas and widget compatibility only.
They do not establish real ComfyUI execution, HTTP guide transport, model
conditioning, saved audio or performance. Retain both server and ComfyUI outputs:

```bash
ffprobe -v error -count_frames -show_streams -show_format -of json "$OUTPUT"
ffmpeg -v error -i "$OUTPUT" -f null -
```
