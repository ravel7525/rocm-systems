# RCCL Release-Gate — Launch Architecture (Our Strategy)
### The v1 validation stack we designed

Posture: **release-gate** — answer "is this RCCL artifact ship-ready?" on
customer-representative AI workloads.
Cluster: mia1 · gfx950 / MI355X · AMD AINIC Pollara RoCE (rdma0–7) · SLURM `amd-tw` / `rccl_dev`

---

## 1. What this validates

```mermaid
mindmap
  root((RCCL ship-ready?))
    Real workloads
      training primus Llama
      inference sglang disagg
      external MLPerf anchor
    All collective directions
      AllReduce BW + latency
      RS+AG small + large
      All-to-All MoE
      P2P pipeline
    Pinned everything-else
      banner-SHA gate
      sticky-node lock
      n>=5 repeatability
    Two KPI families
      RCCL collectives
      AINIC fabric KV
```

The thesis: a single A/B (RCCL build C vs baseline B) with **everything but
librccl pinned**, run across enough cells that every collective "direction" a
customer exercises is lit up by *some* real model.

---

## 2. The stack (layered)

```mermaid
flowchart TB
    subgraph L5["WORKLOAD — backends"]
        W1["primus / Megatron-LM<br/>training"]
        W2["sglang disagg xPyD<br/>inference (MoRI A2A + Mooncake KV)"]
        W3["MLPerf harness<br/>external anchor"]
    end
    subgraph L4["RCCL OVERLAY — the unit under test"]
        O1["build librccl @ SHA inside base image"]
        O2["cp -L over /opt/rocm/lib/librccl.so.1.0.*"]
        O3["patchelf --add-needed librocm_smi64.so.1"]
        O4["banner: strings | grep develop:&lt;sha7&gt;"]
        O1 --> O2 --> O3 --> O4
    end
    subgraph L3["IMAGE — base"]
        I1["rocm/pytorch-private sglang-*-mi35x-mori"]
        I2["primus megatron base"]
    end
    subgraph L2["FABRIC + SCHED"]
        F1["AINIC RoCE rdma0-7 · ionic · GDRDMA"]
        F2["SLURM amd-tw / rccl_dev · sticky nodelist"]
    end
    subgraph L1["HW"]
        H1["MI355X gfx950 · 8 GPU/node"]
    end
    L5 --> L4 --> L3 --> L2 --> L1

    style L5 fill:#3d5afe,color:#fff
    style L4 fill:#e53935,color:#fff
    style L3 fill:#f59e0b,color:#fff
    style L2 fill:#009688,color:#fff
    style L1 fill:#1a1a2e,color:#fff
```

The RCCL overlay (L4) is the *only* thing that changes between build B and C.
Every other layer is byte-identical — that is what makes the delta attributable.

---

## 2b. Backends — the RCCL integration axis (first-class)

The backend is **not** a detail of the workload layer — it is its own coverage
axis. The *same model* drives RCCL differently per backend because each has its
own torch.distributed integration, its own collective call sequence, and its own
**overlay injection point**. A regression can hit one backend's RCCL path and
miss another's.

```mermaid
flowchart TB
    subgraph PRIMUS["primus / Megatron-LM — TRAINING"]
        pb1["parallelism: TP · PP · SP · (FSDP)"]
        pb2["collectives: AllReduce BW · RS+AG · P2P"]
        pb3["overlay inject: cp -L → torch/lib/ (bundled)"]
        pb4["launcher: torchrun in alloc"]
    end
    subgraph SGLANG["sglang disagg xPyD — INFERENCE"]
        sb1["parallelism: TP · EP · DP-attention"]
        sb2["collectives: AllReduce LL · All-to-All (MoRI)"]
        sb3["KV: Mooncake (NOT RCCL)"]
        sb4["overlay inject: cp -L → /opt/rocm/lib + patchelf rsmi"]
        sb5["launcher: sglang.launch_server + router"]
    end
    subgraph MLPERF["MLPerf harness — ANCHOR"]
        mb1["fixed reference config / rules"]
        mb2["collectives: as the reference model dictates"]
        mb3["overlay inject: per base image"]
    end

    style PRIMUS fill:#e8f5f3,stroke:#009688
    style SGLANG fill:#eef0ff,stroke:#3d5afe
    style MLPERF fill:#fff4e6,stroke:#f59e0b
```

**Why the injection point differs per backend** (the load-bearing detail):

| Backend | torch librccl resolution | Inject target | Extra step |
|---------|--------------------------|---------------|------------|
| primus | torch DT_RPATH `$ORIGIN` → bundled `torch/lib/librccl.so` | `…/torch/lib/` | — |
| sglang | `ctypes.find_library("rccl")` → `/opt/rocm/lib/librccl.so.1.0.*` | ROCm system lib | `patchelf --add-needed librocm_smi64.so.1` (else `import torch` dies on `rsmi_init`) |
| vLLM (v2) | TBD — verify at S0 (may be either) | banner-gate decides | TBD |

> The overlay recipe is **per backend**, not universal. Reusing primus's
> bundled-`torch/lib` recipe on sglang silently loads the stock RCCL (banner-gate
> catches it; the `rsmi_init` import-death catches the patchelf miss).

```mermaid
flowchart LR
    M["model weights"] --> B1["primus"] --> R1["RCCL path A<br/>(training collectives)"]
    M --> B2["sglang"] --> R2["RCCL path B<br/>(MoRI A2A + decode LL)"]
    M -.v2.-> B3["vLLM"] -.-> R3["RCCL path C<br/>(distinct integration)"]

    style B1 fill:#009688,color:#fff
    style B2 fill:#3d5afe,color:#fff
    style B3 fill:#888,color:#fff
    style R3 fill:#fdecea,stroke:#e53935
```

This is exactly why **vLLM is a deferred coverage gap, not a redundant model
list** — it is a *different RCCL codepath* for the same models.

---

## 3. The cell matrix (v1)

```mermaid
flowchart LR
    subgraph TR["TRAINING — primus"]
        C1["C1 · Llama-3.1 8B<br/>4 nodes · BF16+FP8"]
        C2["C2 · Llama-3.1 70B + PP=4<br/>4-8 nodes · BF16"]
        C3["C3 · 70B FSDP (if flag)<br/>4-8 nodes · BF16"]
    end
    subgraph IN["INFERENCE — sglang disagg 1P+1D · 5 nodes · FP8"]
        C4["C4 · Llama-3.1 8B<br/>dense"]
        C5["C5 · DeepSeek-V3.2<br/>MoE EP=16"]
        C6["C6 · Llama-4 Scout<br/>MoE"]
    end
    subgraph AN["ANCHOR — MLPerf"]
        C7["C7 · training 8B<br/>8 nodes"]
        C8["C8 · inference 70B-99<br/>1 node"]
    end

    style TR fill:#e8f5f3,stroke:#009688
    style IN fill:#eef0ff,stroke:#3d5afe
    style AN fill:#fff4e6,stroke:#f59e0b
```

Inference cells carry two orthogonal axes:

| Axis | Values | Lights up |
|------|--------|-----------|
| `traffic` | `uniform` / `realistic` (sharegpt-dist) | deterministic baseline / **P5 ragged collectives** |
| `ctx` | `short` (1k/128) / `long` (32k/4k) | small-msg / **long-message latency** |

---

## 4. RCCL pattern coverage

```mermaid
flowchart TB
    P1["P1 AllReduce BW-bound<br/>prefill / training"] --> C1 & C2 & C4 & C5 & C6
    P2["P2 AllReduce latency-bound<br/>decode"] --> C4 & C5 & C6
    RSAGS["RS+AG small (TP-SP)"] --> C1 & C2
    RSAGL["RS+AG large (FSDP)"] -.if flag.-> C3
    P4["P4 All-to-All MoE"] --> C5 & C6
    P6["P6 Pipeline P2P"] --> C2
    P5["P5 ragged collectives"] --> CR["C4-C6 realistic traffic"]
    KV["KV transfer<br/>(Mooncake — NOT RCCL)"] -.separate KPI.-> C4 & C5 & C6

    style P1 fill:#3d5afe,color:#fff
    style P2 fill:#3d5afe,color:#fff
    style P4 fill:#e53935,color:#fff
    style P6 fill:#009688,color:#fff
    style P5 fill:#f59e0b,color:#fff
    style KV fill:#888,color:#fff
```

---

## 5. Launch flow (one gate cycle)

```mermaid
flowchart LR
    B0["resolve pair<br/>B=last good · C=develop HEAD"] --> B1
    B1["build 2× overlay images<br/>(B,C) + build-time SHA assert"] --> B2
    B2["distribute to sticky nodes<br/>docker save | ssh load"] --> B3
    B3["acquire sticky-lock<br/>pinned nodelist"] --> B4
    B4["per cell: 1-iter/1-prompt SANITY"] -->|fail| STOP["abort cell"]
    B4 -->|pass| B5["banner-SHA gate<br/>grep NCCL INFO RCCL version"]
    B5 -->|mismatch| STOP
    B5 -->|match| B6["run n>=5 · phase-aware parse"]
    B6 --> B7["aggregate delta C vs B"]
    B7 -->|all |Δ| in noise| PASS["certify → C becomes next B"]
    B7 -->|regression| FILE["block + bisect → PR"]

    style B5 fill:#e53935,color:#fff
    style PASS fill:#009688,color:#fff
    style FILE fill:#f59e0b,color:#fff
    style STOP fill:#1a1a2e,color:#fff
```

---

## 6. Integrity gates (the 95% of the job)

| Gate | Failure it prevents | Source lesson |
|------|--------------------|----------------|
| **Banner-SHA** | overlay didn't load → A vs A, 0% "reproducible" | ca8ec1f fiasco |
| **patchelf librocm_smi64** | `import torch` dies `undefined symbol: rsmi_init` | sglang overlay |
| **Sticky-node lock** | different nodes → 1-2% topology swing masquerades as RCCL delta | cliff 4 |
| **1-iter/1-prompt sanity** | link breaks / MoRI OOM / graph-capture fail caught in 60s | cliffs 3,5 |
| **Phase-aware parse** | BF16+FP8 in one perf.csv → last-write wins, silent data loss | primus BF16 trigger |
| **n≥5** | noise mistaken for signal (proven ±0.04-0.06% repeatable) | develop-vs-drop |

---

## 7. KPI layering

```mermaid
flowchart LR
    subgraph RCCL["RCCL-attributable"]
        K1["training tok/s/gpu · TFLOPS/gpu"]
        K2["inference TTFT · TPOT/ITL · throughput"]
        K3["MoE A2A step-fraction"]
    end
    subgraph FAB["AINIC fabric (shared, not RCCL collectives)"]
        K4["Mooncake KV bytes · MC_TE_METRIC"]
        K5["NIC counters rdma0-7"]
    end

    style RCCL fill:#eef0ff,stroke:#3d5afe
    style FAB fill:#f2f4f8,stroke:#888
```

---

## 8. Scope boundary (v1)

```mermaid
flowchart TB
    subgraph IN_["IN v1"]
        i1["MoE + dense inference (sglang)"]
        i2["dense training + PP + FSDP"]
        i3["MLPerf anchors"]
        i4["ragged + long-context variants"]
    end
    subgraph V2["DEFERRED v2"]
        d1["vLLM 2nd backend"]
        d2["long-stability 24h"]
        d3["RLHF / DPO"]
        d4["large scale 32+ nodes"]
    end
    subgraph OUT["EXCLUDED (justified)"]
        o1["fine-tune (= pretrain RCCL surface)"]
        o2["multimodal (≈ text-only allreduce)"]
        o3["fault-tolerance (synthetic, not our scope)"]
        o4["MoE training (= 70B dense ⊕ MoE-inf A2A)"]
    end

    style IN_ fill:#e8f5f3,stroke:#009688
    style V2 fill:#fff4e6,stroke:#f59e0b
    style OUT fill:#fdecea,stroke:#e53935
```

**Open item before this becomes a true gate:** a formal pass/fail criterion
(threshold %, metric, n, statistic) — the matrix exists, the decision rule does not yet
