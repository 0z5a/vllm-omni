import site
site.addsitedir('/mnt/d/0z5a/flux-ovis-consolidation-20261001/dependencies')
import pytest
raise SystemExit(pytest.main(['-s','-o','addopts=','tests/diffusion/model_loader/test_diffusers_loader.py','-k','stream_online_quant_weights or process_weights_skips_completed_online_quant_layer or offloaded_dit_load_device_follows_its_component_quantization']))
