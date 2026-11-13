# Working on OrbitWeaver

Start with README.md, docs/method.md, docs/data.md and docs/backends.md.

- Keep rank placement inside each job's GPU allocation.
- Preserve graph dependencies, recorded compute durations, arrivals and physical links.
- Deliver the best reference-validated plan and retain every evaluation attempt in the cost ledger.
- Test invalid-plan accounting, cache accounting, complete execution and collective participant matching.
- Generate test workloads programmatically and keep local trace products under data/ or runs/.
- Keep English and Chinese README sections aligned; describe implemented capabilities directly.
- Preserve Chakra attribution and the Apache-2.0 notices when changing the schema adapter.

Validation:
```bash
python -m unittest discover -s tests -v
orbitweaver demo --output runs/demo
```

The core planner is in src/orbitweaver/planner.py; structured agent tools are in
agent.py and client.py. Data preparation lives in src/orbitweaver/data/.
