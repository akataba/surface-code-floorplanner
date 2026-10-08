#!/usr/bin/env python3
"""Build a pinned upstream checkout; requires platform development libraries."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

PIN = "82a1cbe64717a64a289dd4b0d02e3dc38cae3890"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix", default="", help="CMake dependency prefix list, separated by semicolons"
    )
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not shutil.which("cmake"):
        parser.error("Install cmake, e.g. python -m pip install cmake pkgconf")
    checkout = ROOT / ".deps/liblsqecc"
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/latticesurgery-com/liblsqecc.git", str(checkout)],
            check=True,
        )
        subprocess.run(["git", "-C", str(checkout), "checkout", PIN], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "submodule", "update", "--init", "--recursive"], check=True
        )
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        actual != PIN
        or subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain"], text=True
        ).strip()
    ):
        parser.error("Existing checkout must be clean and at pinned revision; no files were reset")
    command = [
        "cmake",
        "-S",
        str(checkout),
        "-B",
        str(checkout / "build"),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    ]
    if args.prefix:
        command.append("-DCMAKE_PREFIX_PATH=" + args.prefix)
    pkg = shutil.which("pkg-config") or shutil.which("pkgconf")
    if pkg:
        command.append("-DPKG_CONFIG_EXECUTABLE=" + pkg)
    subprocess.run(command, check=True)
    subprocess.run(
        [
            "cmake",
            "--build",
            str(checkout / "build"),
            "--target",
            "lsqecc_slicer",
            "-j",
            str(args.jobs),
        ],
        check=True,
    )
    (ROOT / ".local.json").write_text(
        json.dumps({"binary": str(checkout / "build/lsqecc_slicer"), "commit": PIN}, indent=2)
        + "\n"
    )
    print("Compiler ready. Run python -m lattice_floorplan --help from the repository.")


if __name__ == "__main__":
    main()
