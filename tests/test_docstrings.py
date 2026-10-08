"""Regression checks for typed documentation coverage failures."""

import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "docstring_checker", Path(__file__).resolve().parents[1] / "scripts/check_docstrings.py"
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

VALID = '''"""Example module."""
def convert(value, *, scale=1):
    """Scale a value.

    Args:
        value (float): Value to scale.
        scale (float): Positive multiplier.

    Returns:
        float: Scaled value.

    Raises:
        ValueError: Scale is not positive.
    """
    if scale <= 0:
        raise ValueError('scale')
    return value * scale
'''


class DocumentationContractTests(unittest.TestCase):
    def test_complete_contract(self):
        self.assertEqual(CHECKER.check_source(VALID, "example.py"), [])

    def test_parameter_types_and_names_are_required(self):
        for replacement in ("value: Value to scale.", "other (float): Value to scale."):
            with self.subTest(replacement=replacement):
                broken = VALID.replace("value (float): Value to scale.", replacement)
                self.assertTrue(
                    any("Args mismatch" in e for e in CHECKER.check_source(broken, "x"))
                )

    def test_return_and_exception_sections_are_required(self):
        for before, after, diagnostic in (
            ("Returns:", "Result:", "Returns"),
            ("ValueError: Scale", "RuntimeError: Scale", "Raises"),
        ):
            with self.subTest(diagnostic=diagnostic):
                errors = CHECKER.check_source(VALID.replace(before, after), "x")
                self.assertTrue(any(diagnostic in e for e in errors))

    def test_nested_return_does_not_belong_to_parent(self):
        source = '''"""Example."""
def outer():
    """Create an unused closure."""
    def inner():
        """Return a count.

        Returns:
            int: The count.
        """
        return 1
'''
        self.assertEqual(CHECKER.check_source(source, "nested"), [])

    def test_dataclass_fields_need_typed_attributes(self):
        source = '''"""Example."""
from dataclasses import dataclass
@dataclass
class Settings:
    """Settings."""
    count: int = 1
'''
        self.assertTrue(any("Attributes mismatch" in e for e in CHECKER.check_source(source, "x")))
