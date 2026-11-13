# Sources and acknowledgements

## MLCommons Chakra

The execution-trace schema `et_def.proto` and generated `et_def_pb2.py` in
`src/orbitweaver/_vendor/` originate from [MLCommons Chakra](https://github.com/mlcommons/chakra),
licensed under Apache-2.0. The bundled generated definitions target protobuf 6.31.1.
The complete license is provided in [LICENSE](LICENSE).

The H200 processing workflow uses the source traces and companion conversion tools
from the [MLCommons Chakra MLSys 2026 artifact](https://doi.org/10.5281/zenodo.19636323).
Dataset provenance and processing steps are documented in [docs/data.md](docs/data.md).

## Simulation interfaces

The command backend provides an integration point for execution-graph simulators,
including [ASTRA-sim](https://github.com/astra-sim/astra-sim).
OrbitWeaver's bundled analysis and event-replay modules implement directed-path
demand, dependency readiness, and collective-level resource admission in Python.
