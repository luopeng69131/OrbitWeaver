"""Read length-framed Chakra ET records and match collective participants."""
from pathlib import Path
import ast
import json
import re
from .._vendor import et_def_pb2 as et
from .common import digest, normalize

OPERATIONS = {0: "allreduce", 2: "allgather", 5: "broadcast",
              6: "alltoall", 7: "reducescatter", 8: "reducescatter", 9: "barrier"}


def messages(path):
    with Path(path).open("rb") as stream:
        first = True
        while True:
            byte = stream.read(1)
            if not byte:
                if first:
                    raise ValueError("Empty Chakra ET stream")
                return
            size, shift = 0, 0
            while True:
                size |= (byte[0] & 127) << shift
                if byte[0] < 128:
                    break
                shift += 7
                if shift > 63:
                    raise ValueError("Invalid protobuf length framing")
                byte = stream.read(1)
                if not byte:
                    raise ValueError("Truncated protobuf length")
            if size > 256 * 1024 * 1024:
                raise ValueError("Chakra ET record exceeds the framing limit")
            raw = stream.read(size)
            if len(raw) != size:
                raise ValueError("Truncated protobuf record")
            node = et.GlobalMetadata() if first else et.Node()
            node.ParseFromString(raw)
            if not first:
                yield node
            first = False


def rank_number(path):
    found = re.findall(r"\d+", Path(path).stem)
    if not found:
        raise ValueError("ET filenames must end with their rank ID")
    return int(found[-1])


def load_chakra(directory):
    paths = sorted(Path(directory).glob("*.et"), key=rank_number)
    if not paths or [rank_number(p) for p in paths] != list(range(len(paths))):
        raise ValueError("Provide contiguous per-rank Chakra ET files")
    groups, rows = {}, []
    for rank, path in enumerate(paths):
        for index, node in enumerate(messages(path)):
            attrs = {a.name: getattr(a, a.WhichOneof("value")) for a in node.attr if a.WhichOneof("value")}
            if "process_group:init" in node.name and node.inputs.values:
                wrapper = ast.literal_eval(node.inputs.values)
                if wrapper:
                    decoded = json.loads(wrapper[0]) if isinstance(wrapper[0], str) else wrapper[0]
                    for group in decoded:
                        members = tuple(group["ranks"] or range(len(paths)))
                        key = str(group["pg_name"])
                        if key in groups and groups[key] != members:
                            raise ValueError("Conflicting process-group membership: " + key)
                        groups[key] = members
            if node.type == et.COMM_COLL_NODE:
                values = ast.literal_eval(node.inputs.values) if node.inputs.values else []
                sequence = attrs.get("comm_tag", values[1] if len(values) > 1 else None)
                if sequence is None:
                    raise ValueError("Collective identity requires a sequence or communication tag")
                pg = str(attrs.get("pg_name", "0"))
                typ = int(attrs["comm_type"])
                if typ not in OPERATIONS:
                    raise ValueError("Unsupported Chakra collective type: " + str(typ))
                size = int(attrs["comm_size"])
                group_size = len(groups[pg]) if pg in groups else int(values[-1]) if values else len(paths)
                identity = "collective:" + json.dumps([pg, sequence, typ, size], sort_keys=True, separators=(",", ":"))
                kind = "collective"
            else:
                if node.type not in (et.COMP_NODE, et.METADATA_NODE, et.INVALID_NODE):
                    raise ValueError("Unsupported Chakra execution node type: " + str(node.type))
                kind = "metadata" if node.type in (et.METADATA_NODE, et.INVALID_NODE) else "cpu" if attrs.get("is_cpu_op", False) else "compute"
                identity, size, group_size, typ = "", 0, 1, 0
            rows.append(dict(rank=rank, id=str(node.id), kind=kind, deps=list(node.data_deps),
                             duration_s=node.duration_micros * 1e-6 if kind != "metadata" else 0.0,
                             identity=identity, bytes=size, group_size=group_size,
                             operation=OPERATIONS.get(typ, "allreduce"), source_order=index,
                             metadata={"name": node.name, "control_dependencies": list(node.ctrl_deps)}))
    return normalize(rows, len(paths), {"source": "MLCommons Chakra execution traces",
                     "dataset_doi": "10.5281/zenodo.19636323",
                     "sources": [{"file": p.name, "sha256": digest(p)} for p in paths]})
