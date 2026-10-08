"""Fail-closed subprocess adapter for the pinned machine-output contract."""

import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from .layout import Layout

PIN = "82a1cbe64717a64a289dd4b0d02e3dc38cae3890"


def digest(data):
    """Return the hexadecimal SHA-256 fingerprint of a bytes-like input.

    Args:
        data (bytes | bytearray | memoryview): Bytes to fingerprint; strings must be encoded
            before calling.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class CompilerConfig:
    """Immutable compiler switches included in geometry-cache identity.

    ``distillation_time`` is the deterministic supply replenishment interval
    in compiler timesteps; ``timeout`` is an external wall-clock limit in seconds.
    ``local`` enables instruction decomposition independently of ``pipeline``.
    The legacy default is False: pass ``local=True`` for current benchmarks.
    These settings do not model physical factory failures or QEC cycles.

    Attributes:
        pipeline (str): 'wave' or 'stream'; default 'wave'.
        router (str): 'graph_search' or 'graph_search_cached'.
        graph_search (str): 'djikstra' (upstream spelling) or 'astar'.
        distillation_time (int): Positive deterministic replenishment interval; default 10
            compiler timesteps.
        staggered (bool): Enable staggered supply timing; default True.
        timeout (float): Positive finite wall-clock seconds per native subprocess.
        local (bool): Enable local decomposition; legacy default False, explicitly set True for
            current evaluations.
    """

    pipeline: str = "wave"
    router: str = "graph_search"
    graph_search: str = "djikstra"
    distillation_time: int = 10
    staggered: bool = True
    timeout: float = 15.0
    local: bool = False

    def __post_init__(self):
        """Reject unsupported routing/pipeline choices and invalid timing settings.

        Raises:
            ValueError: Unsupported pipeline/router/search mode, nonpositive replenishment
                interval, or nonpositive/nonfinite timeout.
        """
        if (
            self.pipeline not in ("stream", "wave")
            or self.router not in ("graph_search", "graph_search_cached")
            or self.graph_search not in ("djikstra", "astar")
        ):
            raise ValueError("Unsupported backend configuration")
        if self.distillation_time < 1 or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Timing settings must be positive and finite")


class Compiler:
    """Invoke a supplied slicer binary and persist auditable, fail-closed records."""

    def __init__(self, binary, config=None):
        """Resolve and hash an existing executable path, using config or legacy defaults.

        Raises ValueError if the file is absent. Its hash records provenance;
        it does not certify that the executable was built from the pinned source.

        Args:
            binary (str | os.PathLike[str]): Existing native slicer executable path; the file is
                hashed at construction.
            config (CompilerConfig | None): Backend settings; None uses the legacy defaults,
                including local=False.

        Raises:
            ValueError: The binary path is not a file.
            OSError: The binary cannot be read for hashing.
        """
        self.binary = Path(binary).resolve()
        if not self.binary.is_file():
            raise ValueError(f"Compiler does not exist: {self.binary}")
        self.config = config or CompilerConfig()
        self.binary_sha256 = digest(self.binary.read_bytes())

    def run(
        self,
        circuit_text,
        layout,
        directory,
        input_format="lli",
        retain_slices=True,
        preprocess_layout=None,
    ):
        """Compile one circuit into a new evidence directory and return its record.

        Creates directory and writes input, stdout/stderr, result.json and optional
        slices.json. The directory must not already exist. Summary-only counting
        uses the pinned wave counter minus one. Area is tiles; slices are compiled
        frames, not physical QEC cycles. Partial output is never a valid score.
        Timeout handling kills and reaps the direct child, not a process tree.

        Args:
            circuit_text (str): Circuit source in the input format accepted by the caller; no
                synthesis is performed.
            layout (Layout | str): Explicit ASCII layout or native generator name compact,
                compact_no_clogging or edpc.
            directory (str | os.PathLike[str]): Destination directory for the artifacts
                described above.
            input_format (str): Input syntax, either 'lli' or 'qasm'; default 'lli'.
            retain_slices (bool): Whether to save full JSON frames; False requires the pinned
                wave summary convention.
            preprocess_layout (str | None): Adapter preprocessing mode (compact,
                compact_no_clogging, edpc), or None for raw layout input.

        Returns:
            dict[str, Any]: Status/reason, slices, instructions, resources, hashes, command and
                artifact_directory. Failed/timeout records have slices=None; only success is a
                usable score.

        Raises:
            ValueError: Format, layout or preprocessing/counting combination is unsupported.
            OSError: Evidence paths cannot be created/read/written, including an existing output
                directory. Invocation errors inside the subprocess block are returned as failure
                records.
        """
        if input_format not in ("lli", "qasm"):
            raise ValueError("Expected lli or qasm")
        if preprocess_layout is not None and (
            not isinstance(layout, Layout)
            or preprocess_layout not in ("compact", "compact_no_clogging", "edpc")
        ):
            raise ValueError(
                "Geometry preprocessing requires an ASCII Layout and named native mode"
            )
        if not retain_slices and self.config.pipeline != "wave":
            raise ValueError("Summary-only counting is supported only for the pinned wave pipeline")
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        circuit_path = directory / ("input." + input_format)
        circuit_path.write_text(circuit_text)
        output = directory / "slices.json"
        cfg = self.config
        command = [
            str(self.binary),
            "-i",
            str(circuit_path),
            "-I",
            input_format,
            "-o",
            str(output),
            "-f",
            "machine",
            "-P",
            cfg.pipeline,
            "-r",
            cfg.router,
            "-g",
            cfg.graph_search,
            "--disttime",
            str(cfg.distillation_time),
            "--graceful",
        ]
        if cfg.local:
            command += ["--local"]
        if not retain_slices:
            command += ["--noslices"]
        if not cfg.staggered:
            command += ["--nostagger"]
        resources = None
        if isinstance(layout, Layout):
            (directory / "layout.txt").write_text(layout.text)
            if preprocess_layout is None:
                command += ["-l", str(directory / "layout.txt")]
            else:
                command += [
                    "-L",
                    preprocess_layout,
                    "--geometry-file",
                    str(directory / "layout.txt"),
                ]
            resources = layout.resources()
            layout_sha = digest(layout.text.encode())
        elif layout in ("compact", "compact_no_clogging", "edpc"):
            command += ["-L", layout]
            layout_sha = None
        else:
            raise ValueError("Layout must be a Layout instance or built-in generator name")
        started = time.monotonic()
        status, reason, count, instructions, seconds, code = "error", "", None, None, None, None
        stdout, stderr = "", ""
        reported_count = None
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, timeout=cfg.timeout, cwd=directory
            )
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
            match = re.fullmatch(r"\s*(\d+),(\d+),([\d.eE+\-]+)\s*", stdout)
            if code != 0:
                reason = "Compiler returned nonzero status"
            elif not match or stderr.strip():
                reason = "Missing clean machine summary or compiler diagnostics present"
            else:
                instructions, reported_count, seconds = (
                    int(match[1]),
                    int(match[2]),
                    float(match[3]),
                )
                if not math.isfinite(seconds):
                    raise ValueError("Nonfinite compiler time")
                if not retain_slices:
                    count = reported_count - 1
                    if count < 1:
                        raise ValueError("No compiled frames")
                else:
                    slices = json.loads(output.read_text())
                    if not isinstance(slices, list) or not slices or not math.isfinite(seconds):
                        raise ValueError("Slice JSON is empty or invalid")
                    count = len(slices)
                    # Pinned wave counter starts at 1, increments after every frame.
                    # Stream retry frames use different accounting.
                    if cfg.pipeline == "wave" and reported_count != count + 1:
                        raise ValueError("Wave summary must equal emitted frames plus one")
                    height, width = len(slices[0]), len(slices[0][0])
                    if (
                        not height
                        or not width
                        or any(
                            not isinstance(s, list)
                            or len(s) != height
                            or any(not isinstance(r, list) or len(r) != width for r in s)
                            for s in slices
                        )
                    ):
                        raise ValueError("Malformed/nonrectangular slice grid")
                    if resources and (height, width) != (resources["height"], resources["width"]):
                        raise ValueError("Slice dimensions differ from input layout")
                    if not resources:
                        resources = {
                            "height": height,
                            "width": width,
                            "area": height * width,
                            "counts": None,
                            "note": "Generated layout: tile resources not inferred from dynamic slices",
                        }
                status = "success"
        except subprocess.TimeoutExpired as exc:
            status, reason = "timeout", "External wall-clock timeout; partial output is not a score"
            stdout = (
                (exc.stdout or b"").decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                (exc.stderr or b"").decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
        except (OSError, ValueError, IndexError, TypeError) as exc:
            reason = str(exc)
        result = {
            "status": status,
            "reason": reason,
            "slices": count if status == "success" else None,
            "instructions": instructions if status == "success" else None,
            "reported_slices": reported_count if status == "success" else None,
            "metric": "compiled slice frames; not physical QEC cycles",
            "count_source": "json_frames" if retain_slices else "pinned_wave_counter_minus_one",
            "retained_slices": retain_slices,
            "compiler_seconds": seconds if status == "success" else None,
            "wall_seconds": time.monotonic() - started,
            "returncode": code,
            "resources": resources,
            "command": command,
            "config": asdict(cfg),
            "input_sha256": digest(circuit_text.encode()),
            "layout_sha256": layout_sha,
            "binary_sha256": self.binary_sha256,
            "artifact_directory": str(directory),
        }
        (directory / "stdout.txt").write_text(stdout)
        (directory / "stderr.txt").write_text(stderr)
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
