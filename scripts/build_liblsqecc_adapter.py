#!/usr/bin/env python3
"""Link a geometry-only adapter against an existing pinned native build."""

import argparse, subprocess, shlex, json, hashlib
from pathlib import Path
from lattice_floorplan.compiler import PIN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--build", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    src = Path(a.source).resolve()
    build = Path(a.build).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (
        subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
        != PIN
    ):
        raise ValueError("Wrong native compiler pin")
    flags = (build / "CMakeFiles/lsqecclib.dir/flags.make").read_text()
    includes = shlex.split(
        next(x.split("=", 1)[1] for x in flags.splitlines() if x.startswith("CXX_INCLUDES ="))
    )
    link = shlex.split((build / "CMakeFiles/lsqecc_slicer.dir/link.txt").read_text())
    libs = link[link.index("liblsqecclib.a") + 1 :]
    archive = build / "liblsqecclib.a"
    root = Path(__file__).resolve().parents[1]
    objects = []
    commands = []

    def run(cmd):
        commands.append(cmd)
        subprocess.run(cmd, check=True)

    for kind in ["edpc", "compact"]:
        obj = out / (kind + ".o")
        name = "make_" + kind + "_layout"
        source = src / "src/layout/dynamic_layouts" / (kind + "_layout.cpp")
        run(
            [
                "c++",
                "-std=c++20",
                "-O2",
                *includes,
                "-D" + name + "=original_" + name,
                "-c",
                str(source),
                "-o",
                str(obj),
            ]
        )
        objects.append(str(obj))
    run(
        [
            "c++",
            "-std=c++20",
            "-O2",
            *includes,
            str(root / "native/liblsqecc_adapter.cpp"),
            *objects,
            str(archive),
            *libs,
            "-o",
            str(out / "liblsqecc_adapter"),
        ]
    )
    run(
        [
            "c++",
            "-std=c++20",
            "-O2",
            *includes,
            str(root / "native/export_layout.cpp"),
            str(archive),
            *libs,
            "-o",
            str(out / "export_layout"),
        ]
    )
    files = [
        root / "native/liblsqecc_adapter.cpp",
        root / "native/export_layout.cpp",
        archive,
        out / "liblsqecc_adapter",
        out / "export_layout",
        src / "src/layout/dynamic_layouts/edpc_layout.cpp",
        src / "src/layout/dynamic_layouts/compact_layout.cpp",
    ]
    (out / "build.json").write_text(
        json.dumps(
            {
                "native_pin": PIN,
                "commands": commands,
                "sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            },
            indent=2,
        )
    )
    print(out / "liblsqecc_adapter")


if __name__ == "__main__":
    main()
