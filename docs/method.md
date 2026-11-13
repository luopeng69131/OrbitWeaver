# Joint planning with execution feedback

OrbitWeaver plans a rank-to-endpoint permutation and a communication-priority rule
for a collection of concurrent GPU jobs.

## Planning object

A `Problem` contains an execution DAG, disjoint GPU allocations, original job
arrivals and directed optical links. Computation nodes retain recorded durations.
Collective nodes contain the logical members, operation and data volume.
A `Plan` permutes ranks inside each allocation and selects a priority rule and
finite parameters.

## Structural tools

`analysis.features` combines routed collective demand with reverse-DAG remaining
paths. `analysis.summarize` groups communication by job, membership and reserved
links. The agent can inspect these groups before changing placement or order.

The tunable progress rule is

```text
priority =
    (1 - beta) * (remaining_tail - job_span)
    - beta * normalized_source_order * mean_window_span
    - alpha * original_arrival
```

Higher priorities dispatch first. The rule library also provides service-time,
remaining-tail, normalized-tail, job-order and source-order policies.

## Simulation feedback and selection

1. Initialize a feasible plan and obtain its reference result.
2. Present the allocation, path-demand summary and incumbent to the agent.
3. Analyze candidate placements and request fast or reference evaluations.
4. Update the incumbent when a reference-evaluated candidate improves mean JCT.
5. Save the final plan, request history and cost ledger.

Each evaluation consumes budget before plan validation. Failed attempts and cache
hits are recorded alongside successful evaluations. Initial reference evaluation
is included in the configured reference budget. Analysis calls record wall time.
Codex sessions additionally record model process time and reported token usage.

JCT is completion time minus the original job arrival, averaged across jobs.
Placement changes preserve the GPU allocation; priorities affect resource
admission as dependencies become ready.

## Code map

| Module | Role |
|---|---|
| `model.py` | Execution graph, physical links, valid joint plans |
| `analysis.py` | Subgroup/path summaries and dependency-progress priorities |
| `planner.py` | Budgets, candidate evaluation, reference incumbent and ledger |
| `agent.py` / `client.py` | Codex session and structured local tools |
| `backends.py` | Fast estimates, event replay and command integration |
| `data/` | Trace normalization, provenance and scenario construction |
