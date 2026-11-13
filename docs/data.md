# Data sources and processing

## H200: MLCommons Chakra / NeMo Mixtral

Source: **Sridharan et al., MLCommons Chakra MLSys 2026 artifact**, v1,
published April 17, 2026.
[Dataset record](https://zenodo.org/records/19636323) ·
[DOI: 10.5281/zenodo.19636323](https://doi.org/10.5281/zenodo.19636323) ·
[Chakra project](https://github.com/mlcommons/chakra).

Obtain `nemo-chakra-mixtral-8x7B-traces.zip` and the companion
`chakra-main.zip` from the artifact record. Install the companion converter
following its `mlsys26` instructions.

The source trace archive has MD5
`5f9dc7b56da5fae15b18efaf965b7b74`.
OrbitWeaver verifies this identifier and records SHA-256 provenance for the
archive and each extracted rank file. Archive detection uses file contents and
supports the ZIP and compressed-TAR encodings.

```bash
orbitweaver prepare-h200   --archive data/downloads/nemo-chakra-mixtral-8x7B-traces.zip   --output data/h200

orbitweaver convert-chakra   --input data/h200/et   --output data/h200_graph.json
```

The processing sequence is:

1. Extract `host_0.json`–`host_7.json` and `device_0.json`–`device_7.json`.
2. Run `chakra_trace_link` for each host/device pair.
3. Run `chakra_converter PyTorch` to produce per-rank Chakra ET streams.
4. Parse length-framed protobuf records; retain compute, CPU and metadata nodes.
5. Match collective records by process group, sequence, operation and byte size.
6. Validate participant coverage and dependency connectivity.

Replay follows ET `data_deps`. Control-dependency IDs are retained in node
metadata. Collective timing is derived from operation size, placement, routed
links and the selected evaluation backend.

## A100: measured synchronous-training segments

The [released A100 dataset](../datasets/a100) contains both workloads:
eight original rank files per workload, with 16 measured iterations after three
warmup iterations. The collection date is October 1, 2026; hardware/software
metadata and per-file SHA-256 values are in the [dataset manifest](../datasets/a100/manifest.json).

Process the supplied records:

```bash
orbitweaver convert-a100 \
  --input datasets/a100/transformer --output data/a100_transformer_graph.json
```

The acquisition program `scripts/capture_a100.py` records synchronized data-parallel
training with CUDA events. The Transformer uses 12 layers, hidden size 768,
12 attention heads, feedforward dimension 3,072, batch size 4 and sequence length
512 per rank. ResNet-18 uses batch size 16 per rank and 224×224 inputs.

The program initializes model weights and inputs with reproducible seeds,
performs three warmup iterations, and records 16 measured iterations by default.
Each rank file includes hardware/software metadata, model dimensions, collective
sizes, sequence IDs, computation durations and collection timestamps. The original
study collected its A100 workloads on October 1, 2026.

Install a CUDA-enabled PyTorch environment and collect a new workload:

```bash
torchrun --standalone --nproc_per_node=8 scripts/capture_a100.py   --model transformer --out data/a100_transformer

orbitweaver convert-a100   --input data/a100_transformer   --output data/a100_transformer_graph.json
```

Use `--model resnet18` to capture the other workload.

## Concurrent scenarios

```bash
orbitweaver build-case   --graph data/h200_graph.json   --jobs 4   --condition slow-links   --bandwidth-gbps 50   --arrival-gap-s 0.01   --seed 131000   --output data/h200_case.json
```

Whole-window replay assigns two windows to even-numbered jobs and one to
odd-numbered jobs by default. Rank-local terminal dependencies connect successive
windows. Each job receives its own GPU allocation.

| Condition | Allocation | Arrivals | Capacity |
|---|---|---|---|
| `contiguous` | Contiguous endpoint blocks | Simultaneous | Nominal |
| `fragmented` | Seeded fragmented sets | Simultaneous | Nominal |
| `staggered` | Seeded fragmented sets | Configured gap | Nominal |
| `slow-links` | Seeded fragmented sets | Configured gap | Selected links at 40% / 60% |

`repeat_jobs` accepts an explicit repetition vector for custom workloads.
The graph, source-file checksums and scenario configuration travel together in
the prepared problem JSON.

Run prepared problems with [the simulator interface](simulator.md).
