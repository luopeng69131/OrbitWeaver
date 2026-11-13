# OrbitWeaver A100 synchronous-training traces

Measured Transformer and ResNet-18 execution segments collected on **eight
NVIDIA A100-SXM4-80GB GPUs on October 1, 2026**.

| Workload | Ranks | Measured iterations | Records per rank | Collectives per rank |
|---|---:|---:|---:|---:|
| Transformer | 8 | 16 | 800 | 256 |
| ResNet-18 | 8 | 16 | 128 | 32 |

Each capture follows three warmup iterations. CUDA events measure
forward/backward computation, gradient packing/unpacking, all-reduce and
optimizer updates. The training program uses seeded model weights and synthetic
inputs, BF16 autocast and FP32 gradients.

## Files

- `transformer/rank_0.json`–`rank_7.json`
- `resnet18/rank_0.json`–`rank_7.json`
- [manifest.json](manifest.json): per-file SHA-256, collection timestamps,
  model parameters, recording configuration and preprocessing fingerprints.

The JSON files retain the original measured records and capture metadata.
Compute durations and dependencies form the execution graph; collective byte
sizes and membership drive communication replay on the chosen optical network.

## Use the traces

From the repository root:

```bash
orbitweaver build-simulator --output build/native
orbitweaver convert-a100 --input datasets/a100/transformer --output data/transformer_graph.json
orbitweaver build-case --graph data/transformer_graph.json --jobs 4   --condition slow-links --output data/transformer_case.json
orbitweaver simulate --problem data/transformer_case.json   --binary-directory build/native --output runs/transformer
```

Use `datasets/a100/resnet18` for ResNet-18. The acquisition program is
[`scripts/capture_a100.py`](../../scripts/capture_a100.py); complete acquisition
and processing details are in [docs/data.md](../../docs/data.md).

## Input consistency

The manifest records the compact eight-rank graph produced from each workload.
Checksums validate the raw records, while graph fingerprints track the
dependency-preserving preprocessing output.

## Credit and license

Collected by **Pengcheng Luo and Bowen Zhang** for OrbitWeaver.
Data release: **OrbitWeaver A100 Synchronous-Training Traces, version 1.0 (2026)**.
Licensed under [Apache-2.0](../../LICENSE).
