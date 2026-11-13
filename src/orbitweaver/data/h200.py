"""Extract the pinned NeMo trace artifact and invoke the Chakra converters."""
from pathlib import Path
import json
import shutil
import subprocess
import tarfile
import time
import zipfile
from .common import digest

DATASET = {"title": "MLCommons Chakra MLSys 2026 artifact",
           "url": "https://zenodo.org/records/19636323",
           "doi": "10.5281/zenodo.19636323",
           "archive": "nemo-chakra-mixtral-8x7B-traces.zip",
           "md5": "5f9dc7b56da5fae15b18efaf965b7b74"}


def extract_mixtral(archive, output, expected_md5=DATASET["md5"]):
    archive, output = Path(archive), Path(output)
    if expected_md5 and digest(archive, "md5") != expected_md5:
        raise ValueError("The source archive checksum does not match the dataset record")
    output.mkdir(parents=True, exist_ok=True)
    expected = {kind + "_" + str(rank) + ".json" for kind in ("host", "device") for rank in range(8)}
    seen = set()
    def copy(name, stream):
        basename = Path(name).name
        if basename not in expected:
            return
        if basename in seen:
            raise ValueError("Duplicate trace filename in archive: " + basename)
        seen.add(basename)
        target = output / basename
        if target.exists():
            raise FileExistsError(str(target))
        with target.open("wb") as destination:
            shutil.copyfileobj(stream, destination)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                if not member.is_dir() and Path(member.filename).name in expected:
                    with source.open(member) as stream:
                        copy(member.filename, stream)
    else:
        # Detect compression from bytes: the published filename also occurs as tar.gz.
        with tarfile.open(archive, "r|*") as source:
            for member in source:
                if member.isfile() and Path(member.name).name in expected:
                    with source.extractfile(member) as stream:
                        copy(member.name, stream)
    if seen != expected:
        raise ValueError("Expected host/device traces for all eight ranks")
    record = {**DATASET, "archive_sha256": digest(archive),
              "files": {name: digest(output / name) for name in sorted(seen)}}
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def convert_host_device(raw, output, linker="chakra_trace_link", converter="chakra_converter"):
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for rank in range(8):
        linked = output / ("linked_rank%d.json" % rank)
        target = output / ("chakra_trace.%d.et" % rank)
        commands = [
            ([linker, "--chakra-host-trace", str(raw / ("host_%d.json" % rank)),
              "--chakra-device-trace", str(raw / ("device_%d.json" % rank)), "--rank", str(rank),
              "--output-file", str(linked) + ".tmp"], linked),
            ([converter, "PyTorch", "--input", str(linked), "--output", str(target) + ".tmp"], target),
        ]
        for command, destination in commands:
            if destination.exists():
                raise FileExistsError(str(destination))
            started = time.perf_counter()
            result = subprocess.run(command, capture_output=True, text=True)
            records.append({"rank": rank, "command": command, "exit_code": result.returncode,
                            "wall_s": time.perf_counter() - started})
            (output / "conversion_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError(result.stderr[-2000:])
            temp = Path(str(destination) + ".tmp")
            if not temp.is_file() or temp.stat().st_size == 0:
                raise ValueError("Converter returned an empty trace")
            temp.replace(destination)
    return records
