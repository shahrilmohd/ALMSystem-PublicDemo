"""
Unit tests for ai_layer.knowledge_base.code_loader.

Coverage
--------
- get_engine_code_text returns a non-empty string
- Output contains markdown headers for each file (### path)
- Output wraps code in python fenced blocks
- Missing file paths produce a [File not found] notice, not an exception
- list_engine_files returns a non-empty list of strings
- All default files are relative paths (no absolute paths in the list)
- max_lines_per_file truncates long files and appends a truncation note
- Requesting a specific subset returns only those files
"""
from __future__ import annotations

import os
import textwrap
from unittest.mock import mock_open, patch

import pytest

from ai_layer.knowledge_base.code_loader import (
    _DEFAULT_ENGINE_FILES,
    get_engine_code_text,
    list_engine_files,
)


class TestListEngineFiles:
    def test_returns_non_empty_list(self):
        files = list_engine_files()
        assert len(files) > 0

    def test_all_entries_are_strings(self):
        for f in list_engine_files():
            assert isinstance(f, str)

    def test_no_absolute_paths(self):
        for f in list_engine_files():
            assert not os.path.isabs(f), f"Expected relative path, got: {f}"

    def test_uses_forward_slashes(self):
        for f in list_engine_files():
            assert "/" in f, f"Expected forward slashes in: {f}"

    def test_returns_copy(self):
        a = list_engine_files()
        b = list_engine_files()
        assert a is not b
        assert a == b


class TestGetEngineCodeText:
    def test_returns_non_empty_string(self):
        # Uses real filesystem — at least the header should be present.
        text = get_engine_code_text()
        assert isinstance(text, str)
        assert len(text) > 100

    def test_contains_section_header(self):
        text = get_engine_code_text()
        assert "## Engine source code" in text

    def test_missing_file_produces_not_found_notice(self):
        text = get_engine_code_text(files=["engine/nonexistent_file.py"])
        assert "[File not found]" in text
        assert "engine/nonexistent_file.py" in text

    def test_missing_file_does_not_raise(self):
        # Should not raise FileNotFoundError — graceful degradation.
        get_engine_code_text(files=["this/does/not/exist.py"])

    def test_subset_returns_only_requested_files(self):
        text = get_engine_code_text(files=["engine/core/fund.py"])
        assert "engine/core/fund.py" in text
        # A file not in the subset must not appear.
        assert "engine/results/tvog_calculator.py" not in text

    def test_python_fenced_block_present_for_existing_file(self):
        # fund.py must exist — check it gets wrapped in ```python ... ```
        text = get_engine_code_text(files=["engine/core/fund.py"])
        if "[File not found]" not in text:
            assert "```python" in text

    def test_file_header_uses_markdown_h3(self):
        text = get_engine_code_text(files=["engine/core/fund.py"])
        assert "### engine/core/fund.py" in text

    def test_truncation_applied_when_max_lines_exceeded(self):
        # Create a fake file with 10 lines, then request max_lines_per_file=3.
        fake_content = "\n".join(f"line {i}" for i in range(10)) + "\n"

        with patch(
            "builtins.open",
            mock_open(read_data=fake_content),
        ):
            # Also patch os.path.join so it doesn't matter what path we use.
            text = get_engine_code_text(
                files=["engine/fake.py"],
                max_lines_per_file=3,
            )

        assert "lines truncated" in text
        assert "line 0" in text   # First line should be present
        assert "line 9" not in text  # Last line should be truncated

    def test_no_truncation_when_within_limit(self):
        fake_content = "line 1\nline 2\nline 3\n"

        with patch("builtins.open", mock_open(read_data=fake_content)):
            text = get_engine_code_text(
                files=["engine/fake.py"],
                max_lines_per_file=100,
            )

        assert "truncated" not in text
        assert "line 3" in text

    def test_empty_files_list_returns_header_only(self):
        text = get_engine_code_text(files=[])
        assert "## Engine source code" in text
        assert "```python" not in text
