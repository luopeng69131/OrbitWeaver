# Simulation backends

The planning controller calls `evaluate(problem, plan)` and receives mean/per-job
JCT, execution audits and optional diagnostics. This contract makes backend
selection independent of candidate construction.

## Included evaluators

**Native C++ simulator.** Replays the compact execution graph with synchronized
collective rounds and physical resource admission. The companion feature
extractor supplies communication service and remaining-tail values. Build and
API examples are in [simulator.md](simulator.md).

**Critical-path evaluator.** Runs a topological traversal using recorded compute
durations and routed collective service demand. It supplies inexpensive
placement-sensitive estimates.

**Event-replay evaluator.** Processes dependency readiness, GPU/CPU execution
channels, collective communication channels and directed-link reservations.
An admitted collective holds its path links until completion; collectives with
disjoint resources progress concurrently. The output includes per-collective
ready/start/finish times and waits.

The optical network uses directed minimum-hop paths. Bidirectional-ring edges
are enumerated clockwise first, which resolves equal-hop choices consistently.
A 50-Gb/s directional link has a capacity of 6.25e9 B/s. Default spacing is 300 m,
with propagation delay plus a 1-microsecond terminal delay per hop.

Collective service combines maximum routed link demand/capacity with
round-dependent path latency. Supported operations are all-reduce, all-gather,
reduce-scatter, all-to-all, broadcast and barrier. Implementations are in
`network.py` and `backends.py`.

## External evaluator contract

A command backend receives a single JSON object on standard input:

```json
{"problem": {"schema_version": 1, "...": "graph and physical configuration"},
 "plan": {"placement": [0, 1], "priority": {"rule": "progress", "alpha": 0, "beta": 0}}}
```

It writes a JSON result on standard output:

```json
{"mean_jct_s": 1.5, "jct_s": [1.0, 2.0],
 "audits": {"complete_graph": true, "resources_released": true}}
```

The adapter checks per-job coverage, finite values, the mean and execution audits.
Evaluator processes receive explicit argument lists. Their errors and runtime
are included in the controller's request records.

Exercise this interface with the bundled JSON bridge:

```bash
orbitweaver run-agent   --problem runs/demo/problem.json   --output runs/command-agent   --fast-command '["python", "-m", "orbitweaver.bridge", "--kernel", "critical-path"]'   --reference-command '["python", "-m", "orbitweaver.bridge", "--kernel", "event-replay"]'
```

An [ASTRA-sim](https://github.com/astra-sim/astra-sim) integration can translate the
problem graph, rank mapping and communication-priority plan into its native
inputs, then return complete-execution JCT and the associated audits through
the same JSON contract.
