# Orbital GPU cluster simulator

The native simulator executes a measured dependency graph on an optically
interconnected GPU formation. It models CPU/GPU execution channels, collective
readiness, routed communication rounds, endpoint/link reservations and job
arrival times. Its event kernel and feature extractor originate from the
OrbitWeaver experimental simulator.

## Build and run

A C++17 compiler builds the two executables, `orbit_fast` and `orbit_features`.

```bash
orbitweaver build-simulator --output build/native
orbitweaver convert-a100 \
  --input datasets/a100/resnet18 --output data/resnet18_graph.json
orbitweaver build-case --graph data/resnet18_graph.json \
  --jobs 2 --condition fragmented --output data/resnet18_case.json
orbitweaver simulate --problem data/resnet18_case.json \
  --binary-directory build/native --diagnostics --output runs/resnet18
```

`result.json` includes mean and per-job JCT, completion counts and execution
audits. The native run directory stores network/priority inputs, feature tables,
process records and optional communication events. These outputs support
examining path sharing, readiness and waits.

## Configure the problem

```bash
orbitweaver build-case --graph data/resnet18_graph.json \
  --jobs 4 --condition slow-links --bandwidth-gbps 200 \
  --arrival-gap-s 0.02 --seed 42 --output data/custom_case.json
```

The supported conditions combine contiguous or fragmented allocation,
simultaneous or staggered arrivals, and selected reduced-capacity links.
Each node represents one GPU endpoint with two neighboring optical connections.
Directional capacity is expressed in Gb/s at the CLI and bytes/s internally.

Eight-rank source workloads can instantiate 16, 32 or 64 endpoints through
whole-window job replay. Dependencies connect successive windows. Compute
durations remain attached to the measured nodes; communication timing follows
the configured physical interconnect.

## Placement and communication order

A plan maps logical ranks to their allocated physical GPUs and chooses a
priority rule. Load a saved plan with `--plan`, or select a rule directly:

```bash
orbitweaver simulate --problem data/custom_case.json \
  --rule progress --alpha -1 --beta -10 \
  --binary-directory build/native --output runs/custom_priority
```

Develop a new scheduling algorithm through the Python API:

```python
from orbitweaver import Plan
from orbitweaver.native.backend import NativeBackend

backend = NativeBackend("build/native", "runs/my_scheduler", diagnostics=True)
plan = Plan(tuple(my_rank_placement))
scores = {node.id: my_priority(node) for node in problem.graph.nodes
          if node.kind == "collective"}
result = backend.evaluate(problem, plan, scores=scores)
```

The adapter validates allocation-preserving placement and complete, finite
collective priorities. JCT is completion minus the original arrival.
The compact graph format preserves rank participation counts and checks
complete execution.

## Extending the simulator

- `native/fast.cpp`: event dispatch, resource occupancy and collective rounds.
- `native/features.cpp`: native communication service and reverse-DAG remaining paths.
- `native/backend.py`: compilation, compact export, inputs, execution audits and metrics.
- `data/scenarios.py`: allocations, arrival patterns and optical-link configurations.
- `backends.py`: Python evaluators and the external-simulator command contract.

The native format supports up to 64 logical ranks. New network models can expose
their directed links through `Problem.links`; evaluation adapters can implement
the same `evaluate(problem, plan)` interface.
