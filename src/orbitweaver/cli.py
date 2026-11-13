"""Command-line entry points for graph processing and agent planning."""
from pathlib import Path
import argparse
import json
from .model import Graph, Plan, Problem
from .backends import CriticalPathBackend, EventReplayBackend, CommandBackend
from .planner import Planner


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="OrbitWeaver joint planning and trace preparation")
    sub = parser.add_subparsers(dest="action", required=True)
    demo = sub.add_parser("demo", help="Generate a workload and exercise the planning tools")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    for command in ("convert-a100", "convert-chakra"):
        p = sub.add_parser(command)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare-h200", help="Extract the dataset artifact and run Chakra linking/conversion")
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
    elif args.action == "run-agent":
        from .agent import run_codex
        problem = Problem.from_dict(json.loads(args.problem.read_text()))
        fast = CommandBackend(json.loads(args.fast_command)) if args.fast_command else CriticalPathBackend()
        reference = CommandBackend(json.loads(args.reference_command)) if args.reference_command else EventReplayBackend()
        planner = Planner(problem, fast, reference, args.output, args.fast_budget, args.reference_budget)
        result = run_codex(planner, args.output, args.model, args.effort, args.wall_limit_s)
        print(json.dumps({"best_mean_jct_s": result["best"]["result"]["mean_jct_s"], "agent": result["agent"]}, indent=2))


if __name__ == "__main__":
    main()
