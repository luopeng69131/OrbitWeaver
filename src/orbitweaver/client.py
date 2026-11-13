"""Small structured-tool client used inside an agent workspace."""
import json
import socket


class Client:
    def __init__(self, address):
        self.address = address

    def call(self, operation, **arguments):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(3600)
            connection.connect(self.address)
            connection.sendall((json.dumps({"op": operation, **arguments}) + "\n").encode())
            response = json.loads(connection.makefile("rb").readline(16 << 20))
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["value"]

    def budget(self):
        return self.call("budget")

    def best(self):
        return self.call("best")

    def analyze(self, placements):
        return self.call("analyze", placements=placements)

    def probe(self, placement, rule="progress", kernel="fast", alpha=0, beta=0):
        return self.probe_plan({"placement": list(placement),
                               "priority": {"rule": rule, "alpha": alpha, "beta": beta}}, kernel)

    def probe_plan(self, plan, kernel="fast"):
        return self.call("probe", plan=plan, kernel=kernel)
