import inspect
import textwrap
import vllm.v1.attention.backends.flashinfer as fi
source = textwrap.dedent(inspect.getsource(fi.FlashInferMetadataBuilder.build))
old = "seq_lens_cpu = common_attn_metadata.seq_lens_cpu"
assert source.count(old) == 1
source = source.replace(old, old + "\n            if self.use_dcp:\n                seq_lens_cpu = seq_lens_cpu.clone()")
marker = "    # Adjust num_block_np for cascade attention"
assert source.count(marker) == 1
source = source.replace(marker, "        seq_lens_np = seq_lens_cpu.numpy()\n        num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size\n\n" + marker)
namespace = dict(fi.__dict__)
exec(compile(source, '<diagnostic-dcp-local-kv-metadata>', 'exec'), namespace)
fi.FlashInferMetadataBuilder.build = namespace['build']
