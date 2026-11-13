"""Use the same structural and simulation tools exposed to the agent."""
from orbitweaver import Plan, Planner
from orbitweaver.backends import CriticalPathBackend, EventReplayBackend
from orbitweaver.data import build_problem
from orbitweaver.demo import example_graph

problem = build_problem(example_graph(), jobs=2, condition="fragmented")
planner = Planner(problem, CriticalPathBackend(), EventReplayBackend(), "runs/python-example")
placement = list(problem.initial_placement)
print(planner.analyze([placement]))
candidate = Plan(tuple(placement), rule="progress", alpha=-1, beta=-10)
planner.probe(candidate.to_dict(), "fast")
planner.probe(candidate.to_dict(), "reference")
print(planner.best())
