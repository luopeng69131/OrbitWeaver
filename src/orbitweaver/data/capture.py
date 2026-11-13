"""Normalize measured per-rank synchronous-training segments."""
from pathlib import Path
import json
from .common import digest, normalize


def load_a100(directory):
    paths = sorted(Path(directory).glob("rank_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    if not paths:
        raise ValueError("Provide per-rank capture JSON files")
    payloads = [json.loads(p.read_text()) for p in paths]
    ranks = len(payloads)
    if sorted(p["rank"] for p in payloads) != list(range(ranks)) or any(p["world_size"] != ranks for p in payloads):
        raise ValueError("Capture rank coverage and world size must agree")
    signatures = [[(r["kind"], r.get("sequence"), r.get("bytes", 0)) for r in p["records"]] for p in payloads]
    if any(s != signatures[0] for s in signatures):
        raise ValueError("Capture records must describe matching training segments")
    records = []
    for data in payloads:
        previous = None
        for index, row in enumerate(data["records"]):
            kind = row["kind"]
            if kind not in ("compute", "collective"):
                raise ValueError("Unknown capture segment kind")
            nid = str(index)
            records.append(dict(rank=data["rank"], id=nid, kind=kind,
                                deps=[] if previous is None else [previous],
                                duration_s=row["duration_us"] * 1e-6,
                                bytes=row.get("bytes", 0), operation="allreduce",
                                identity="collective:0:%s" % row.get("sequence"),
                                group_size=ranks, source_order=index))
            previous = nid
    metadata = {"source": "OrbitWeaver synchronous GPU capture",
                "sources": [{"file": p.name, "sha256": digest(p)} for p in paths],
                "capture": {k: v for k, v in payloads[0].items() if k != "records"}}
    return normalize(records, ranks, metadata)
