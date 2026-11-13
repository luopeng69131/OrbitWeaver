"""JSON stdin/stdout evaluator demonstrating the command-backend contract."""
import argparse
import json
import sys
from .backends import CriticalPathBackend, EventReplayBackend
from .model import Plan, Problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["critical-path", "event-replay", "native"], default="event-replay")
    parser.add_argument("--binary-directory", type=str, default="build/native")
    parser.add_argument("--output", type=str, default="runs/native-bridge")
    args = parser.parse_args()
    request = json.load(sys.stdin)
    problem = Problem.from_dict(request["problem"])
    plan = Plan.from_dict(request["plan"]).validate(problem)
    if args.kernel == "native":
        from .native.backend import NativeBackend
        backend = NativeBackend(args.binary_directory, args.output)
    else:
        backend = CriticalPathBackend() if args.kernel == "critical-path" else EventReplayBackend()
    print(json.dumps(backend.evaluate(problem, plan)))


if __name__ == "__main__":
    main()
