## Motivation.

1. Continously to support more models with more acceleration features [Model x Feature]
2. Feature A x Feature B validation: run compatibility check for A x B [Feature x Feature]

**Call for Help!**

---
### 1. Model x Feature

The following table tracks which models are currently supported by each acceleration method:

✅: already supported, PR attached
🙋: **not supported yet, help wanted!**
⏳: not supported yet, with PR raised
❓:  maybe unnecessary to support it. The benefits are minimal.

- **The tasks labeled with P0 are of higher priority**
- **Quantization** methods including FP8, Int8, NVFP4, etc. This RFC only maintains the support of at least one quantization methods. For detaild support list, check #1854 .

#### ImageGen

| Model | ⚡TeaCache | ⚡Cache-DiT | 🔀SP (Ulysses & Ring) | 🔀CFG-Parallel | 🔀Tensor-Parallel | 🔀HSDP | 💾CPU Offload (Layerwise) | 💾VAE-Patch-Parallel Encode/Decode | 💾Quantization |
|-------|:----------:|:-----------:|:---------------------:|:--------------:|:-----------------:|:------:|:------------------------:|:--------------------:|:--------------:|
| **Bagel** | ✅#848 | ✅#736 | ✅#1903 | ✅#1578 | ✅#1293 | ✅#3150 | ✅#2734  | ✅#3982 /✅#3982 | @lsyyysky  |
| **ERNIE-Image** | ⏳#4239 | ✅#2861| ✅#2861| 🙋 |  ✅#2861| ✅#2861| ✅#2861|  🙋 | 🙋|
| **FLUX.1-dev/FLUX.1-schnell** | ⏳#1243 | ✅#1145 | ⏳#2133 | ✅#1269 | ✅#853 | ✅#1900 | ✅#1486  | 🙋 | ✅ |
| **FLUX.2-klein** | ✅#1234 | ✅#1209 | ✅#1250 | ✅#851 | ✅#973 | ✅#1900 | ✅#1486  | 🙋 | ✅ |
| **FLUX.1-Kontext-dev** | ⏳#1243  | ⏳#4205  | ⏳#2133  | ⏳#2281  | ✅#561 | ✅#561 | ✅#1486 | 🙋/🙋 | ✅#2184  |
| **FLUX.2-dev** | ✅#1871 | ✅#1814 |  ⏳#3244 | ✅#2010 | ✅#1629 | ✅#1900 | 🙋 | 🙋 | ⏳#2517  |
| **GLM-Image** | ⏳#1458 | ✅#1399 | ✅#1983 | ✅#763 | ✅#1918 | ✅#2029 |@yuanheng-zhao  | 🙋/🙋 | ✅#2292  |
| **Hidream-I1-Full** |  🙋 | @Sendoh-code  | @Sendoh-code | 🙋 | ✅#2572  | 🙋 |  @fywc  | 🙋 | 🙋 |
| **HunyuanImage3** | ✅#1927 | ✅#1848 | ✅#2163 | ⏳#1751 | ✅#1085 | @ischencheng (P0) | @xldeng-chn | ⏳#3091/⏳#3091 | ✅#1935 |
| **LongCat-Image** | ✅#1487 | ✅#392 | ✅#721 | ✅#851 | ✅#926 | ⏳#3284  | ✅#2339    | ⏳#1921 | ⏳#2633  |
| **LongCat-Image-Edit** | ✅#1487 | ✅#392 | ✅#721 | ✅#851 | ✅#926 | ⏳#3284  | ✅#2339    | 🙋/⏳#1921 | ⏳#2633  |
| **MammothModa2(T2I)** | 🙋 | 🙋 | 🙋 | 🙋 | 🙋 | 🙋 | @yuanheng-zhao   | 🙋 | 🙋 |
| **Nextstep_1(T2I)** | ❓ | ❓ | 🙋(P0) | ✅#612 | ✅#612 | 🙋 | ✅#2339    | 🙋 | 🙋 |
| **OmniGen2** |⏳ #2257 | ✅ | ✅#3206 | ✅#2423 | @zzhuoxin1508 | @yangjianjuan | @yangjianjuan | 🙋/🙋 | ⏳#2441  |
| **Ovis-Image** | ⏳#2332  | ✅ | @yangjianjuan  | ✅#851 | ⏳#4516| 🙋 | ✅#2339   | ⏳#1921 |  ⏳#4516| |
| **Qwen-Image** | ✅#179 | ✅ | ✅#779 | ✅#851 | ✅#830 | ✅#2029 | ✅#858 | ✅#1366 | ✅ |
| **Qwen-Image-2512** | ✅#179 | ✅ | ✅#779 | ✅#851 | ✅#830 | ✅#2029 | ✅#858 | ✅#1366 | ✅ |
| **Qwen-Image-Edit** | ✅#179 | ✅ | ✅#779 | ✅#851 | ✅#830 | ✅#2029 | ✅#858 | 🙋/ ✅#1366 | ⏳#2608  |
| **Qwen-Image-Edit-2509** | ✅#179 | ✅ | ✅#779 | ✅#851 | ✅#830 | ✅#2029 | ✅#858 | 🙋/ ✅#1366 | ⏳#2608  |
| **Qwen-Image-Layered** | ✅#179 | ✅ | ✅#779 | ✅#851 | ✅#830 | ✅#2029 | ✅#858 | 🙋/ ✅#1366 | 🙋 |
| **SenseNova-U1** | ⏳#4164  | 🙋 | 🙋 | 🙋 | ✅#3319  | 🙋 |  ✅#3319  | 🙋 | 🙋 |
| **Stable-Diffusion3.5** | ⏳#852 | ✅#584 | ⏳#654 | ✅#851 | ✅#1336 | 🙋 | ✅#2339   | ✅#1366 | 🙋 |
| **Z-Image** | ✅#817 | ✅ | ✅#779 | ❓ | ✅#735 (TP=2 only) | ✅#2029 | ✅#1486  | 🙋/✅ | ✅ |



#### VideoGen

| Model | ⚡TeaCache | ⚡Cache-DiT | 🔀SP (Ulysses & Ring) | 🔀CFG-Parallel | 🔀Tensor-Parallel | 🔀HSDP | 💾CPU Offload (Layerwise) | 💾VAE-Patch-Parallel Encode/Decode | 💾Quantization |
|-------|:----------:|:-----------:|:---------------------:|:--------------:|:-----------------:|:------:|:------------------------:|:--------------------:|:--------------:|
| **Wan2.1/Wan2.2** | ⏳#1365  | ✅#1021 | ✅#966 | ✅#851 | ✅#964 | ✅#1339 | ✅#858 | ✅#2368 /✅#1366 | ⏳ |
| **Wan2.1-VACE** | ⏳#1365  | ✅#1885 | ✅#1885 | ✅#1885 | ✅#1885 | ✅#1885 | ✅ #1885| ✅#1885 | ⏳ | 
| **LTX-2** | 🙋 | ✅#841 | ✅#841 | ✅#841 | ✅#841 | ✅#2899  |✅ #2018 | 🙋 /⏳ #2135 | ✅#3700|
| **LTX-2.3** |@Sendoh-code | ✅#2893 | ✅#2893 | 🙋 | ✅#2893 | 🙋 | 🙋 | 🙋 | 🙋 |
| **Helios** | 🙋 | ⏳#2473  | ✅#1604 | ✅#1604 | ✅#1604 | ✅#1604 | ✅#1604 | 🙋/ 🙋 | ⏳ |
| **MagiHuman** |  🙋 | 🙋 | 🙋 | ❓ | ✅#2301 | 🙋 | ✅#2301 | 🙋 | 🙋 |
| **HunyuanVideo-1.5 T2V I2V** | ⏳#2682  | ✅#1516 | ✅#2444  | ✅#1516 | ✅#1516 | ✅#1516 | ✅#1516 | 🙋/⏳#2418 | ✅ |
| **DreamID-Omni** | @MoveCloudROY  | ⏳#3265  | @MoveCloudROY  | ✅#1855 | 🙋 | 🙋 | ✅ #2018  | 🙋/🙋 | ✅#3902  |


#### AudioGen

| Model | ⚡TeaCache | ⚡Cache-DiT | 🔀SP (Ulysses & Ring) | 🔀CFG-Parallel | 🔀Tensor-Parallel | 🔀HSDP | 💾CPU Offload (Layerwise) | 💾VAE-Patch-Parallel Encode/Decode | 💾Quantization |
|-------|:----------:|:-----------:|:---------------------:|:--------------:|:-----------------:|:------:|:------------------------:|:--------------------:|:--------------:|
| **Stable-Audio-Open** | ✅#1314 | ⏳#1341 | ❓ | ❓ | ⏳#1406 | ✅#2982 | ✅#2909  | ⏳#3282 | ✅ |

> Notes:
> 1. Nextstep_1(T2I) does not support cache acceleration methods, like tea_cache or cache_dit;
> 2. `Tongyi-MAI/Z-Image-Turbo` and `princepride/daVinci-MagiHuman` are distilled models with minimal NFEs; CFG-Parallel is not necessary.
> 3. VAE-Patch-Parallel Encode/Decode apply to Image-to-Image and Image-to-Video models; VAE-Patch-Parallel Decode applies to Text-to-Image/Video models.


**How to Contribute?**
1. Raise a PR with the proposed code changes to support a new feature

[How to add a diffiusion model tutorial](https://github.com/vllm-project/vllm-omni/blob/main/docs/contributing/model/adding_diffusion_model.md)
--- 

### 2. Feature x Feature

The current compatibility check for Feature A x Feature B is relied on a preliminary test branch #2083 

|  | ⚡TeaCache | ⚡Cache-DiT | 🔀Ulysses-SP | 🔀Ring-Attn | 🔀CFG-Parallel | 🔀Tensor Parallel | 🔀HSDP | 🔀Expert Parallel | 💾CPU Offloading (Layerwise) | 💾CPU Offloading (Module-wise) | 💾VAE Patch Parallel | 💾FP8 Quant | 🔧LoRA Inference |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **⚡TeaCache** | | | | | | | | | | | | | |
| **⚡Cache-DiT** | ❌ | | | | | | | | | | | | |
| **🔀Ulysses-SP** | ✅ | ✅ | | | | | | | | | | | |
| **🔀Ring-Attn** | ✅ | ✅ | ✅ | | | | | | | | | | |
| **🔀CFG-Parallel** | ✅ | ✅ | ✅ | ✅ | | | | | | | | | |
| **🔀Tensor Parallel** | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | |
| **🔀HSDP** | 🙋 | 🙋 | 🙋 | 🙋 | 🙋 | ❌ | | | | | | | |
| **🔀Expert Parallel** | ❌ | ❌ | 🙋 | 🙋 | 🙋 | 🙋 | 🙋 | | | | | | |
| **💾CPU Offloading (Layerwise)** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅#2021 | ❌ | | | | | |
| **💾CPU Offloading (Module-wise)** | #1880 | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ❌ | | | | |
| **💾VAE Patch Parallel** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | | | |
| **💾FP8 Quant** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ | ✅ | ✅ | ✅ | | |
| **🔧LoRA Inference** | 🙋 | 🙋 | 🙋 | 🙋 | 🙋 | 🙋 | 🙋 | 🙋 | 🙋 | 🙋 | 🙋 | 🙋 | |

> Notes:
> 1. Tensor Parallel and HSDP are not compatible.
> 2. TeaCache and Cache-DiT are not compatible.
> 3. CPU Offloading (Layerwise) and CPU Offloading (Module-wise) are not compatible.
> 4. CPU Offloading (Layerwise) supports single-card for now.
> 5. Using FP8-Quant as an example of qunatization methods.
> 6. Expert Parallel must be used together with other parallelism methods, thus not compatible with teacache or cache-dit alone.

In `docs/user_guide/diffusion_features.md`, it maintains the feature compatibility matrix shown above.

**How to Contribute?**
1. Raise a PR with the test results and propose to update the `docs/user_guide/diffusion_features.md` 
2. If you find the compatibility issue, please raise an issue and ask for help.

### Proposed Change.

This RFC will be maintained continuously, as more models and features will be added in the future.

Comments are welcomed!

### Feedback Period.

_No response_

### CC List.

@hsliuustc0106 @david6666666 @SamitHuang @ZJY0516 @dongbo910220 

### Any Other Things.

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://vllm-omni.readthedocs.io), which can answer lots of frequently asked questions.

