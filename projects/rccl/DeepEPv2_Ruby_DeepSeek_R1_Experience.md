# DeepEPv2 on Ruby/Thor2: Benchmarks and DeepSeek-R1 Serving Experience

Date: 2026-08-18
Hardware: Ruby/Thor2, 8 × gfx950 / MI350X, Broadcom `bnxt_re` RoCE NICs

```mermaid
flowchart TB
BASE["deepep-ruby-poc:local<br/>ROCm 7.14 · Torch 2.11 · RCCL 2.30.4"]
DEEP["DeepEPv2_private @ hipify 2af71dc<br/>setup.py build → _C.so + include/"]
WHEEL["vLLM wheel 455edc02"]
OVR["vllm_overrides/<br/>deepep_v2.py · all2all.py · deepep_ht.py"]
AIT["AITER d9e5ef7c<br/>+ zero-token ksplit patch"]
TV["torchvision text-only stub"]
IMG["vllm-private-deepep:coherent"]
BASE --> IMG
DEEP --> IMG
WHEEL --> IMG
OVR -->|"cp поверх site-packages"| IMG
AIT --> IMG
TV --> IMG
```

```mermaid
flowchart LR
subgraph node["1 узел · 8× MI350X"]
direction TB
subgraph dp["8 DP-движков, TP=1, EP=8"]
R0["rank 0<br/>эксперты 0–31"]
R1["rank 1<br/>эксперты 32–63"]
RN["… rank 7<br/>эксперты 224–255"]
end
EB["ElasticBuffer (V2)<br/>symmetric memory окно"]
R0 --- EB
R1 --- EB
RN --- EB
end
EB -.->|"RCCL GIN IB proxy<br/>bnxt_re0..7"| EB
```

## Executive summary

The native DeepEPv2 1×8 ElasticBuffer benchmark is correct and fast on Ruby.
DeepSeek-R1 model initialization also works on the same hardware after staging
the checkpoint locally. The remaining R1 blocker in the SGLang path is not
model loading or RCCL initialization: it is an incomplete HIP FP8
`deepep_normal -> triton` adapter in SGLang.

The current first alternative is vLLM built with the mandatory private
`AMD-ROCm-Internal/DeepEPv2_private` repository, `hipify` branch. Its
DeepEPV2 modular MoE adapter already implements the per-expert layout that is
missing from SGLang.

## Hardware and base runtime

```mermaid
flowchart LR
  subgraph Ruby["Ruby / Thor2 node"]
    G0["gfx950 GPU 0"]
    G1["gfx950 GPU 1"]
    G2["gfx950 GPU 2"]
    G7["gfx950 GPU 7"]
    IF["xGMI / Infinity Fabric"]
    G0 --- IF
    G1 --- IF
    G2 --- IF
    G7 --- IF
  end

  subgraph Network["Scale-out fabric"]
    N0["bnxt_re0"]
    N7["bnxt_re7"]
  end

  Ruby --- Network
```

Primary validated THERock stack:

| Layer | Version / selection |
|---|---|
| Torch | 2.11.0, HIP 7.14.60850 |
| RCCL | `d22b8646`, overlaid into Torch’s bundled `librccl.so` |
| DeepEP | HIP build, ElasticBuffer V2 |
| GPU target | `gfx950` |
| RDMA userspace | rdma-core v61 in native DeepEP image |
| Model checkpoint | DeepSeek-R1, 163 safetensors, ~642 GB |

The RCCL binary used by Torch and DeepEP was checked by SHA-256 and matched.

## 1×8 native DeepEPv2 benchmark

The benchmark is not a synthetic NCCL all-reduce. It directly creates
`deep_ep.ElasticBuffer`, dispatches routed tokens, combines the expert outputs,
and checks both phases against DeepEP reference implementations.

```mermaid
sequenceDiagram
  participant R as 8 torch.distributed ranks
  participant D as DeepEP init_dist
  participant E as ElasticBuffer
  participant K as HIP dispatch/combine kernels
  participant Ref as DeepEP reference implementation

  R->>D: warm device, init NCCL, create fresh group(range(8))
  D->>E: ElasticBuffer(tokens=4096, hidden=7168, topk=8)
  R->>K: dispatch(remote-round-robin routing)
  K-->>R: received expert-token layout + handle
  R->>Ref: compare dispatch source indices
  R->>K: combine(expert output, handle)
  K-->>R: combined token output
  R->>Ref: bitwise combine correctness check
```

Configuration:

| Parameter | Value |
|---|---:|
| EP ranks | 8 |
| Tokens per rank | 4096 |
| Hidden dimension | 7168 |
| Top-k | 8 |
| Experts | 256 |
| Data type | BF16 |
| Warmup / timed iterations | 50 / 50 |

Results:

| Backend | Correctness | Dispatch GB/s | Combine GB/s |
|---|---|---:|---:|
| DeepEPv2 | PASS on all 8 ranks | 186.271 | 87.307 |
| MoRI reference | PASS on all 8 ranks | 253.100 | 293.660 |

The MoRI result is a functional reference, not a fair performance winner:
DeepEP used THERock ROCm 7.14 / Torch 2.11, while the usable MoRI reference
used a separate SGLang ROCm 7.2 / Torch 2.9 image.

## Why the benchmark works but initial R1 serving did not

```mermaid
flowchart TB
  subgraph Bench["Native benchmark: PASS"]
    B1["DeepEP init_dist"]
    B2["GPU warm-up"]
    B3["Fresh NCCL group(range(8))"]
    B4["ElasticBuffer dispatch/combine"]
    B1 --> B2 --> B3 --> B4
  end

  subgraph OldServe["Original SGLang R1 path: FAIL"]
    S1["SGLang internal TP ProcessGroup"]
    S2["backend._comm_ptr()"]
    S3["ElasticBuffer topology query"]
    S4["Private ncclComm field dereference"]
    S5["SIGSEGV"]
    S1 --> S2 --> S3 --> S4 --> S5
  end

  Bench -. "different communicator lifecycle" .-> OldServe
```

The SGLang adapter lazily constructs ElasticBuffer during the first MoE
forward. The original DeepEP HIP code passed PyTorch’s private communicator
pointer into a topology query that read RCCL implementation fields
(`ncclTeamWorld`, `ncclTeamLsa`). On this Torch 2.11 ProcessGroup, the pointer
was null or incompatible for that use.

The following focused fixes advanced execution:

1. `EP_FORCE_RDMA_RANKS=1`, `EP_FORCE_NVL_RANKS=8` bypasses private topology
   field reads for the 1×8 topology only.
2. DeepEP rejects a null `_comm_ptr()` and creates a managed communicator.
3. `SGLANG_DEEPEP_PRECREATE_COMM=1` creates that communicator before model/KV
   allocation, avoiding late CUMEM bootstrap allocation under memory pressure.
4. `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256` preserves the explicit
   bounded capacity instead of replacing it with worker fallback 16384.
5. `SGLANG_SCHEDULER_SKIP_ALL_GATHER=1` avoids SGLang’s unnecessary single-node
   CPU all-gather path.

With these changes, ElasticBuffer successfully created its 30 MB GPU buffer,
created d22 NCCL device communicators, and started DeepEP dispatch on all
ranks.

## DeepSeek-R1 model startup lessons

```mermaid
flowchart LR
  HF["Shared HF cache\n642 GB symlinked snapshot"]
  NFS["Cold-node mmap + H2D"]
  HSA["hsakmt_ioctl stall\naround 97 GB/GPU"]
  Scratch["/scratch/ilkosare/DeepSeek-R1\nlocal NVMe copy"]
  Load["All 163 safetensors loaded"]
  KV["KV cache allocated\n~115-126 GB/GPU"]
  HTTP["HTTP server ready"]

  HF --> NFS --> HSA
  HF --> Scratch --> Load --> KV --> HTTP
```

Loading directly from the shared HF cache stalled on cold Ruby nodes in
`hsakmt_ioctl` during FP8 weight copy. `SGLANG_DISABLE_ASYNC_WEIGHT_LOADING=1`
alone was insufficient on cold nodes.

Staging the resolved 642 GB snapshot to local `/scratch` at roughly 800 MB/s
eliminated this issue. With `/model` mounted from local NVMe, R1 consistently:

- loaded all 163 shards;
- allocated the KV cache;
- started its HTTP service.

## Final SGLang blocker

After ElasticBuffer construction, SGLang’s HIP FP8 DeepEP normal path reaches
the local expert compute stage:

```mermaid
flowchart LR
  D["DeepEP V2 dispatch\nper-expert received layout"]
  P["Required adapter:\ndeepep_normal -> runner input"]
  T["Triton FP8 MoE"]
  C["DeepEP combine"]

  D --> P --> T --> C
```

The current SGLang branch has no registered
`deepep_normal -> triton` pre-permute/post-permute implementation:

```text
AssertionError: Pre-permute function for deepep_normal to triton is not registered
```

The standard Triton permutation cannot be used as a substitute. DeepEP normal
dispatch has a per-expert receive layout, invalid expert slots, scatter
indices, and combine metadata. Forcing standard Triton routing triggers a
DeepEP HIP duplicate-expert assertion and an HSA hardware exception.

An AITER adapter does exist, but its gfx950 runtime JIT module (`module_moe_asm`)
segfaulted in this stack. This is why SGLang R1 did not generate a token even
though its model and communication initialization advanced much further.

## Private DeepEPv2 repository

Required source:

```text
https://github.com/AMD-ROCm-Internal/DeepEPv2_private
branch: hipify
commit: 2af71dcd129afe7aa1d5a3c10a869e79771563b6
```

Important observations:

- `main` README is CUDA-first, but `hipify` contains an actual HIP path:
  `EP_TARGET_HIP`, RCCL roots, hipcc/amdclang and `PYTORCH_ROCM_ARCH`.
- The private HIPIFY source compiled and imported successfully on gfx950 /
  HIP 7.14.
- Its own ElasticBuffer test passed a first 8-rank case on Ruby:

```text
Ranks: 1 x 8
Experts: 8 / 256
Tokens: 256 (max: 256), hidden: 7168
QPs: 129 / 129
```

## Current path 1: vLLM + private DeepEPv2

```mermaid
flowchart TB
  M["DeepSeek-R1 checkpoint\nlocal /scratch"]
  V["vLLM ROCm wheel\nbuilt for gfx950"]
  P["DeepEPv2_private/hipify\nEP_TARGET_HIP=1"]
  EP["vLLM DeepEPV2PrepareAndFinalize"]
  R["DP=8 / EP=8 / TP=1\nexpert parallel serving"]
  API["OpenAI-compatible vLLM API"]

  M --> V
  P --> EP
  V --> EP --> R --> API
```

vLLM differs materially from SGLang:

- it has `DeepEPV2PrepareAndFinalize`;
- it implements DeepEP V2 prefill layout (`do_expand=True`,
  `do_cpu_sync=True`);
- it turns received per-expert counts into `ExpertTokensMetadata`;
- it has matching dispatch and combine logic in its modular MoE implementation.

The private DeepEPv2 HIPIFY package passed its native smoke test. A persistent
`vllm-private-deepep:local` image was then built on c07 with:

- private HIPIFY DeepEPv2;
- locally built ROCm vLLM wheel;
- `PYTHONPATH=/opt/rocm/share/amd_smi` so vLLM detects ROCm GPUs;
- broken torchvision removed for text-only R1.

The next vLLM launch uses:

```text
--tensor-parallel-size 1
--data-parallel-size 8
--data-parallel-size-local 8
--enable-expert-parallel
--all2all-backend deepep_high_throughput
--enforce-eager
--gpu-memory-utilization 0.70
```

This is the preferred non-SGLang path because it has an existing DeepEP V2
adapter rather than requiring an unsafe port of the missing SGLang adapter.

## Other non-SGLang directions

```mermaid
flowchart LR
  Goal["DeepSeek-R1 with private DeepEPv2"]
  V["1. vLLM ROCm\ncurrent path"]
  M["2. Megatron-LM V2 adapter\ninference harness"]
  C["3. Custom PyTorch runner\nlast resort"]

  Goal --> V --> M --> C
```

1. **vLLM ROCm + private HIPIFY DeepEPv2**
   Best first path: production-oriented API and existing V2 prepare/finalize
   adapter.

2. **Megatron-LM + V2 adapter**
   Useful for a distributed R1 generation harness and correctness reference.
   It is not a turnkey HTTP serving path; it needs a DeepSeek checkpoint
   loader plus FP8/grouped-MoE execution configuration.

3. **Custom PyTorch runner**
   Maximum control but maximum implementation work. It must recreate
   per-expert scatter/grouped FP8 GEMM/inverse-scatter/combine semantics.
   This is only justified if both vLLM and Megatron integration paths fail.

## Runtime controls used in serving experiments

```bash
export EP_TARGET_HIP=1
export EP_DISABLE_LEGACY=1
export EP_GIN_QUEUE_DEPTH=0
export NCCL_GIN_TYPE=2
export NCCL_CUMEM_ENABLE=1
export NCCL_IB_HCA=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,\
bnxt_re4:1,bnxt_re5:1,bnxt_re6:1,bnxt_re7:1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=fenic0
export GLOO_SOCKET_IFNAME=fenic0
export HSA_NO_SCRATCH_RECLAIM=1
```

`EP_GIN_QUEUE_DEPTH=0` is intentional: RCCL GIN IB proxy requires its backend
default rather than a caller-provided queue depth.

## Current status

| Item | Status |
|---|---|
| Native private HIPIFY DeepEPv2 1×8 smoke | PASS |
| R1 local checkpoint staging | PASS |
| SGLang R1 model loading | PASS |
| SGLang ElasticBuffer/Gin initialization | PASS after targeted fixes |
| SGLang R1 token generation | BLOCKED by missing HIP FP8 adapter |
| vLLM ROCm wheel + private DeepEPv2 image | BUILT |
| vLLM R1 DeepEP serving request | Next execution step |
