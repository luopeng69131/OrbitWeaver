"""Trace processing, simulator execution and agent planning."""
from pathlib import Path
import argparse
import json
from .model import Graph, Plan, Problem, RULES
from .backends import CriticalPathBackend, EventReplayBackend, CommandBackend
from .planner import Planner


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    path.write_text(json.dumps(value, indent=2) + "\n")


def policy_arguments(parser):
    parser.add_argument("--rule", choices=sorted(RULES), default="progress")
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--beta", type=float, default=0)


def backend_arguments(parser):
    parser.add_argument("--backend", choices=["native", "replay"], default="native")
    parser.add_argument("--binary-directory", type=Path, default=Path("build/native"))


def main():
    parser = argparse.ArgumentParser(description="OrbitWeaver data, simulator and joint-planning tools")
    sub = parser.add_subparsers(dest="action", required=True)
    demo = sub.add_parser("demo", help="Generate a workload and exercise the planning tools")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    for command in ("convert-a100", "convert-chakra"):
        p = sub.add_parser(command)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare-h200")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--linker", default="chakra_trace_link")
    p.add_argument("--converter", default="chakra_converter")
    p = sub.add_parser("build-case")
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--condition", choices=["contiguous", "fragmented", "staggered", "slow-links"], default="fragmented")
    p.add_argument("--bandwidth-gbps", type=float, default=50)
    p.add_argument("--arrival-gap-s", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=131000)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("build-simulator", help="Build the native event kernel and feature extractor")
    p.add_argument("--output", type=Path, default=Path("build/native"))
    p.add_argument("--compiler", default="g++")
    p = sub.add_parser("simulate", help="Execute a joint plan on the orbital GPU model")
    p.add_argument("--problem", type=Path, required=True)
    p.add_argument("--plan", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diagnostics", action="store_true")
    policy_arguments(p)
    backend_arguments(p)
    p = sub.add_parser("run-agent")
    p.add_argument("--problem", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="gpt-6.1-sol")
    p.add_argument("--effort", default="xhigh")
    p.add_argument("--wall-limit-s", type=float, default=1800)
    p.add_argument("--fast-budget", type=int, default=64)
    p.add_argument("--reference-budget", type=int, default=8)
    p.add_argument("--fast-command", help="JSON argument list for an external evaluator")
    p.add_argument("--reference-command", help="JSON argument list for an external evaluator")
    p.add_argument("--native-binary-directory", type=Path, help="Use the native fast backend for candidate exploration")
    args = parser.parse_args()
    if args.action == "demo":
        from .demo import run_demo
        result = run_demo(args.output)
        print(json.dumps({"initial_mean_jct_s": result["initial_mean_jct_s"],
                          "best_mean_jct_s": result["best"]["result"]["mean_jct_s"],
                          "remaining": result["remaining"], "output": str(args.output)}, indent=2))
    elif args.action in {"convert-a100", "convert-chakra"}:
        from .data import load_a100, load_chakra
        loader = load_a100 if args.action == "convert-a100" else load_chakra
        graph = loader(args.input)
        save(args.output, graph.to_dict())
        print(json.dumps({"ranks": graph.ranks, "nodes": len(graph.nodes), "output": str(args.output)}))
    elif args.action == "prepare-h200":
        from .data.h200 import extract_mixtral, convert_host_device
        extract_mixtral(args.archive, args.output / "raw")
        convert_host_device(args.output / "raw", args.output / "et", args.linker, args.converter)
        print(json.dumps({"et_directory": str(args.output / "et")}))
    elif args.action == "build-case":
        from .data import build_problem
        graph = Graph.from_dict(json.loads(args.graph.read_text()))
        problem = build_problem(graph, args.jobs, args.condition, args.bandwidth_gbps, args.arrival_gap_s, args.seed)
        save(args.output, problem.to_dict())
        print(json.dumps({"jobs": args.jobs, "ranks": problem.graph.ranks, "output": str(args.output)}))
    elif args.action == "build-simulator":
        from .native.backend import build_simulator
        result = build_simulator(args.output, args.compiler)
        print(json.dumps({"binary_directory": str(args.output), "binaries": [r["target"] for r in result["binaries"]]}))
    elif args.action == "simulate":
        problem = Problem.from_dict(json.loads(args.problem.read_text()))
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Choose a fresh simulation output directory")
        args.output.mkdir(parents=True, exist_ok=True)
        if args.plan:
            data = json.loads(args.plan.read_text())
            if "best" in data:
                data = data["best"]["plan"]
            elif "plan" in data:
                data = data["plan"]
            plan = Plan.from_dict(data).validate(problem)
        else:
            plan = Plan(problem.initial_placement, args.rule, args.alpha, args.beta).validate(problem)
        if args.backend == "native":
            from .native.backend import NativeBackend
            backend = NativeBackend(args.binary_directory, args.output / "native", args.diagnostics)
        else:
            backend = EventReplayBackend()
        result = backend.evaluate(problem, plan)
        save(args.output / "plan.json", plan.to_dict())
        save(args.output / "result.json", result)
        print(json.dumps({"mean_jct_s": result["mean_jct_s"], "jct_s": result["jct_s"],
                          "backend": result["backend"], "audits": result["audits"]}, indent=2))
    elif args.action == "run-agent":
        from .agent import run_codex
        problem = Problem.from_dict(json.loads(args.problem.read_text()))
        if args.fast_command:
            fast = CommandBackend(json.loads(args.fast_command))
        elif args.native_binary_directory:
            from .native.backend import NativeBackend
            fast = NativeBackend(args.native_binary_directory, args.output / "native-fast")
        else:
            fast = CriticalPathBackend()
        reference = CommandBackend(json.loads(args.reference_command)) if args.reference_command else EventReplayBackend()
        planner = Planner(problem, fast, reference, args.output, args.fast_budget, args.reference_budget)
        result = run_codex(planner, args.output, args.model, args.effort, args.wall_limit_s)
        print(json.dumps({"best_mean_jct_s": result["best"]["result"]["mean_jct_s"], "agent": result["agent"]}, indent=2))


if __name__ == "__main__":
    main()
