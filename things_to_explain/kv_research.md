## 10. Reading list, surveys, blogs, videos, and what to skip

Read in this order. It moves from the system you know (HiSparse) to the codecs, then to the closest match for your idea.

| # | Work | Venue / status | Who | Code | Why read it |
| --- | --- | --- | --- | --- | --- |
| 1 | [DSA explainer](https://sebastianraschka.com/llms-from-scratch/ch04/15_deepseek_sparse_attention/) | Blog | Sebastian Raschka | Yes | Gentle background on the indexer |
| 2 | [HiSparse](https://arxiv.org/html/2608.07009) | arXiv 2026; merged in SGLang | Stanford, Meta, NVIDIA Research, Alibaba, Ant | Yes | The baseline system and the miss-rate data |
| 3 | [CacheGen](https://arxiv.org/pdf/2310.07240) | SIGCOMM 2024 | UChicago, MSR, Stanford | [Yes](https://github.com/UChi-JCL/CacheGen) | Delta coding with anchors; codec design rules |
| 4 | [KVTC](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3fb6f10bd2784f6cfb6a6ed6280df40c-Abstract-Conference.html) | ICLR 2026 | NVIDIA | Not checked | Transform coding; shared directions without references |
| 5 | [DeltaKV](https://arxiv.org/html/2602.08005v1) | arXiv 2026, unreviewed | HIT Shenzhen | [Yes](https://github.com/CURRENTF/Sparse-vLLM) | Closest to your idea: residuals plus decode-on-select |
| 6 | [ShadowKV](https://arxiv.org/html/2410.21465v2) | ICML 2025 spotlight | ByteDance Seed, CMU | [Yes](https://github.com/bytedance/ShadowKV) | GPU shadow plus CPU fetch; low-rank keys |
| 7 | [Coupled Quantization](https://arxiv.org/pdf/2405.03917) | NeurIPS 2024 | Rice, Stevens | Not checked | Mutual information between channels, made into a codec |
| 8 | [CommVQ](https://arxiv.org/abs/2506.18879) | ICML 2025 | Apple, UMass | [Yes](https://github.com/UMass-Embodied-AGI/CommVQ) | Additive VQ that works with RoPE |
| 9 | [SpeCache](https://proceedings.mlr.press/v267/jie25a.html) | ICML 2025 | Peking University and co-authors | Not checked | Low-bit GPU copy, full copy on CPU |
| 10 | [PQCache](https://arxiv.org/abs/2407.12820v2) | SIGMOD 2025 | Peking University, Baichuan | Not checked | KV as vector search |
| 11 | [xKV](https://arxiv.org/abs/2503.18893) | arXiv 2025 | Cornell | [Yes](https://github.com/abdelfattah-lab/xKV) | Cross-layer shared subspaces; MLA result |
| 12 | [TensorRT-LLM KV compression](https://nvidia.github.io/TensorRT-LLM/features/kv-cache-compression.html) | Product docs | NVIDIA | Yes | Production FP4 codec at the tier boundary |
| 13 | [ESS](https://arxiv.org/abs/2512.10576) | arXiv 2025 | Baidu | No | Industry offload study for DeepSeek-V3.2 |

### Surveys

Your worry is right: no survey covers similarity-aware coding for sparse-attention offload. The broad ones are still useful maps.

- **Li et al., "A Survey on LLM Acceleration based on KV Cache Management"**, TMLR 2025. Splits the field into token-level, model-level and system-level work. Comes with the [Awesome-KV-Cache-Management](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management) paper list ([arXiv](https://arxiv.org/abs/2412.19442v1)).
- Two HKUST (Guangzhou) PhD qualifying-exam surveys, 2025 and 2026. Not peer-reviewed, but the 2026 one has a clean four-family taxonomy that includes "placement and hierarchical management" ([2026 abstract](https://dsa.hkust-gz.edu.cn/zh/blog/2026/05/28/breaking-the-memory-wall-in-llm-inference-serving-a-survey-of-kv-cachemanagement-techniques/)).

### Blogs worth your time

- [KV Cache Compression and Its Infra Problems](https://research.nvidia.com/labs/eai/blogs/kv-cache-compression-and-its-infra-problems/), Song Han's group at NVIDIA, June 2026. The best beginner read on why lab methods fail in production: FlashAttention never writes attention scores, and paged memory frees nothing until a whole block is empty.
- [HiSparse on the LMSYS blog](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/), April 2026.
- [DeepSeek-V4 in vLLM](https://vllm.ai/blog/2026-04-24-deepseek-v4) and [NVIDIA's DeepSeek-V4 post](https://developer.nvidia.com/blog/build-with-deepseek-v4-using-nvidia-blackwell-and-gpu-accelerated-endpoints/) for CSA and HCA.
- [SemiAnalysis on GLM-5.3 and memory demand](https://newsletter.semianalysis.com/p/sparse-savings-persistent-demand-inside-glm53): the industry view of why DRAM demand stays high.
