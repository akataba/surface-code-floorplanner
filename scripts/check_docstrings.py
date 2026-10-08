#!/usr/bin/env python3
"""Check typed Google-style contracts without importing optional runtime dependencies."""

import ast
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SCRIPTS = ("benchmark_local.py", "report_local.py", "verify_local.py", "check_docstrings.py")


def section(doc, name):
    """Extract the indented body of one Google-style section.

    Args:
        doc (str): Cleaned docstring from ast.get_docstring.
        name (str): Section heading without its colon.

    Returns:
        str: Section text, or an empty string when the section is absent.
    """
    match = re.search(r"^" + re.escape(name) + r":\n((?:[ \t].*\n|\n)*)", doc + "\n", re.M)
    return match.group(1) if match else ""


def local_nodes(node):
    """Yield descendants belonging to this scope, excluding nested definitions.

    Args:
        node (ast.AST): Function or statement whose descendants should be scanned.

    Yields:
        ast.AST: Descendant syntax nodes, excluding nested function/class bodies.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield child
        yield from local_nodes(child)


def check_source(source, filename):
    """Check argument names/types, returned values, raises and dataclass fields.

    Args:
        source (str): Python source text, parsed without importing or executing it.
        filename (str): Display path used in diagnostics.

    Returns:
        list[str]: Contract violations, empty when all checks pass. This checks
            structure and coverage, not whether prose/types are semantically true.

    Raises:
        SyntaxError: The source is not valid Python.
    """
    tree = ast.parse(source)
    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        location = f"{filename}:{getattr(node, 'lineno', 1)}"
        doc = ast.get_docstring(node)
        if not doc:
            errors.append(f"{location}: missing docstring")
            continue
        if isinstance(node, ast.ClassDef):
            is_dataclass = any(
                (isinstance(d, ast.Name) and d.id == "dataclass")
                or (
                    isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Name)
                    and d.func.id == "dataclass"
                )
                for d in node.decorator_list
            )
            if is_dataclass:
                fields = {
                    n.target.id
                    for n in node.body
                    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                }
                documented = set(
                    re.findall(r"^    (\w+) \([^\n]+\):\s+\S", section(doc, "Attributes"), re.M)
                )
                if fields != documented:
                    errors.append(
                        f"{location}: typed Attributes mismatch: {sorted(fields ^ documented)}"
                    )
            continue
        if isinstance(node, ast.Module):
            continue
        args = {
            a.arg
            for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if a.arg not in {"self", "cls"}
        }
        if node.args.vararg:
            args.add("*" + node.args.vararg.arg)
        if node.args.kwarg:
            args.add("**" + node.args.kwarg.arg)
        documented = re.findall(r"^    (\*{0,2}\w+) \([^\n]+\):\s+\S", section(doc, "Args"), re.M)
        if args != set(documented) or len(documented) != len(set(documented)):
            errors.append(f"{location}: typed Args mismatch: {sorted(args ^ set(documented))}")
        descendants = list(local_nodes(node))
        returns_value = any(
            isinstance(n, ast.Return)
            and n.value is not None
            and not (isinstance(n.value, ast.Constant) and n.value.value is None)
            for n in descendants
        )
        yields = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in descendants)
        required_section = "Yields" if yields else ("Returns" if returns_value else None)
        if required_section and not re.search(
            r"^    [^\n:]+:\s+\S", section(doc, required_section), re.M
        ):
            errors.append(f"{location}: missing typed {required_section}")
        raised = {
            n.exc.func.id
            for n in descendants
            if isinstance(n, ast.Raise)
            and isinstance(n.exc, ast.Call)
            and isinstance(n.exc.func, ast.Name)
        }
        documented_raises = set(re.findall(r"^    (\w+):\s+\S", section(doc, "Raises"), re.M))
        if raised - documented_raises:
            errors.append(f"{location}: missing Raises: {sorted(raised - documented_raises)}")
    return errors


def main():
    """Check all package modules and active local tools and print diagnostics.

    Returns:
        int: Zero if contracts pass, one if any documentation violations exist.

    Raises:
        OSError: A source file cannot be read.
        SyntaxError: A checked file is not valid Python.
    """
    paths = sorted((ROOT / "src/lattice_floorplan").glob("*.py"))
    paths += [ROOT / "scripts" / name for name in ACTIVE_SCRIPTS]
    errors = [
        error
        for path in paths
        for error in check_source(path.read_text(), str(path.relative_to(ROOT)))
    ]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Typed Google-style contracts passed for {len(paths)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
