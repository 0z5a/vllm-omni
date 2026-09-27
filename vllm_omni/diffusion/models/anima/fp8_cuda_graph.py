"""Anima sm_120 online-FP8 CFG graphs with reusable conditioning."""

import torch


def install_anima_fp8_cuda_graph(pipeline):
    eager_diffuse = pipeline.diffuse
    states = {}

    def diffuse(prompt_embeds, negative_prompt_embeds, latents, padding_mask, timesteps, do_true_cfg, true_cfg_scale):
        if (
            not do_true_cfg
            or negative_prompt_embeds is None
            or tuple(latents.shape) != (1, 16, 1, 128, 128)
            or tuple(prompt_embeds.shape) != (1, 512, 1024)
            or tuple(negative_prompt_embeds.shape) != (1, 512, 1024)
        ):
            return eager_diffuse(
                prompt_embeds, negative_prompt_embeds, latents, padding_mask, timesteps, do_true_cfg, true_cfg_scale
            )

        transformer = pipeline.transformer
        key = (
            id(transformer),
            tuple(latents.shape),
            tuple(prompt_embeds.shape),
            tuple(negative_prompt_embeds.shape),
            tuple(padding_mask.shape),
        )
        state = states.get(key)
        if state is None:
            state = {
                "hidden": torch.empty_like(latents, dtype=transformer.dtype),
                "timestep": torch.empty_like(timesteps[:1], dtype=transformer.dtype),
                "contexts": (torch.empty_like(prompt_embeds), torch.empty_like(negative_prompt_embeds)),
                "padding": torch.empty_like(padding_mask),
                "kv": {},
                "graphs": [],
                "outputs": [],
                "retained": [],
            }
            states[key] = state

        for dst, src in zip(state["contexts"], (prompt_embeds, negative_prompt_embeds)):
            dst.copy_(src)
        state["padding"].copy_(padding_mask)
        state["hidden"].copy_(latents)
        state["timestep"].copy_(timesteps[0].expand(latents.shape[0]).to(transformer.dtype) / 1000.0)

        projections = [
            projection
            for block in transformer.transformer_blocks
            for projection in (block.attn2.to_k, block.attn2.to_v)
        ]
        projection_forwards = [(projection, projection.forward) for projection in projections]
        for projection, original in projection_forwards:
            slots = state["kv"].setdefault(projection, {})
            for context in state["contexts"]:
                pointer = context.data_ptr()
                computed = original(context)
                if pointer in slots:
                    slots[pointer].copy_(computed)
                else:
                    slots[pointer] = computed

            def cached(x, saved=slots):
                return saved[x.data_ptr()]

            projection.forward = cached

        eager_forward = transformer.forward
        if not state["graphs"]:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    for context in state["contexts"]:
                        eager_forward(
                            hidden_states=state["hidden"],
                            timestep=state["timestep"],
                            encoder_hidden_states=context,
                            padding_mask=state["padding"],
                            return_dict=False,
                        )
            torch.cuda.current_stream().wait_stream(side)

            first = transformer.transformer_blocks[0]
            shared = [
                transformer.patch_embed,
                transformer.rope,
                transformer.time_embed,
                first.norm1,
                first.attn1,
                first.norm2,
                first.attn2.to_q,
                first.attn2.norm_q,
            ]
            if transformer.learnable_pos_embed is not None:
                shared.append(transformer.learnable_pos_embed)
            shared_forwards = []
            for module in shared:
                original = module.forward
                pending = [None]

                def paired(*args, run=original, slot=pending, **kwargs):
                    if slot[0] is None:
                        slot[0] = run(*args, **kwargs)
                        state["retained"].append(slot[0])
                        return slot[0]
                    result = slot[0]
                    slot[0] = None
                    return result

                module.forward = paired
                shared_forwards.append((module, original))

            pool = torch.cuda.graph_pool_handle()
            # The negative graph reads tensors produced by the positive graph.
            for context in state["contexts"]:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    output = eager_forward(
                        hidden_states=state["hidden"],
                        timestep=state["timestep"],
                        encoder_hidden_states=context,
                        padding_mask=state["padding"],
                        return_dict=False,
                    )[0]
                state["graphs"].append(graph)
                state["outputs"].append(output)
            for module, original in shared_forwards:
                module.forward = original

        phase = [0]

        def replay(*, hidden_states, timestep, encoder_hidden_states, padding_mask, return_dict=False):
            assert not return_dict
            branch = phase[0]
            phase[0] = 1 - branch
            state["hidden"].copy_(hidden_states)
            state["timestep"].copy_(timestep)
            state["graphs"][branch].replay()
            return (state["outputs"][branch].clone(),)

        transformer.forward = replay
        try:
            result = eager_diffuse(
                prompt_embeds, negative_prompt_embeds, latents, padding_mask, timesteps, do_true_cfg, true_cfg_scale
            )
        finally:
            transformer.forward = eager_forward
            for projection, original in projection_forwards:
                projection.forward = original
        assert phase[0] == 0
        return result

    pipeline.diffuse = diffuse

    qwen_forward = pipeline._get_qwen_prompt_embeds
    condition_forward = pipeline.condition_prompt_embeds
    cached_key: tuple[int, str, torch.dtype] | None = None
    cached_raw: tuple[torch.Tensor, torch.Tensor] | None = None
    cached_conditioned: torch.Tensor | None = None

    def get_qwen(prompt, max_sequence_length, device, dtype):
        nonlocal cached_key, cached_raw, cached_conditioned
        if prompt == [""] and not pipeline.text_encoder.training and not pipeline.text_conditioner.training:
            key = (max_sequence_length, str(device), dtype)
            if key != cached_key:
                cached_raw = qwen_forward(prompt, max_sequence_length, device, dtype)
                cached_conditioned = None
                cached_key = key
            assert cached_raw is not None
            return cached_raw
        return qwen_forward(prompt, max_sequence_length, device, dtype)

    def condition(
        qwen_prompt_embeds,
        qwen_attention_mask,
        t5_input_ids,
        t5_attention_mask,
        device=None,
        conditioning_dtype=None,
        output_dtype=None,
    ):
        nonlocal cached_conditioned
        if cached_raw is not None and qwen_prompt_embeds is cached_raw[0]:
            if cached_conditioned is None:
                cached_conditioned = condition_forward(
                    qwen_prompt_embeds,
                    qwen_attention_mask,
                    t5_input_ids,
                    t5_attention_mask,
                    device,
                    conditioning_dtype,
                    output_dtype,
                )
            return cached_conditioned
        return condition_forward(
            qwen_prompt_embeds,
            qwen_attention_mask,
            t5_input_ids,
            t5_attention_mask,
            device,
            conditioning_dtype,
            output_dtype,
        )

    pipeline._get_qwen_prompt_embeds = get_qwen
    pipeline.condition_prompt_embeds = condition
