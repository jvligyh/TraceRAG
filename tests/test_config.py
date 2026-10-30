from __future__ import annotations

import os
import tempfile
import unittest
import subprocess
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from tracerag.config import Settings


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class SettingsLocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "project"
        self.other = Path(self.temp.name) / "unrelated"
        self.root.mkdir()
        self.other.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_project_env_and_database_do_not_follow_working_directory(self):
        (self.root / ".env").write_text("TRACERAG_TRANSLATOR=none\n", encoding="utf-8")
        (self.other / ".env").write_text("TRACERAG_TRANSLATOR=deepseek\n", encoding="utf-8")
        with patch("tracerag.config.PROJECT_ROOT", self.root), patch.dict(os.environ, {}, clear=True):
            with working_directory(self.other):
                settings = Settings.from_env()
        self.assertEqual("none", settings.translator)
        self.assertEqual((self.root / "data/runtime").resolve(), settings.data_dir)

    def test_environment_wins_over_project_env_and_relative_data_uses_project(self):
        (self.root / ".env").write_text("TRACERAG_DATA_DIR=from-file\nTRACERAG_TRANSLATOR=marian\n", encoding="utf-8")
        with patch("tracerag.config.PROJECT_ROOT", self.root), patch.dict(os.environ, {
            "TRACERAG_DATA_DIR": "private-library", "TRACERAG_TRANSLATOR": "none"
        }, clear=True), working_directory(self.other):
            settings = Settings.from_env()
        self.assertEqual((self.root / "private-library").resolve(), settings.data_dir)
        self.assertEqual("none", settings.translator)

    def test_absolute_data_path_and_explicit_settings_are_preserved(self):
        absolute = self.other / "library"
        with patch("tracerag.config.PROJECT_ROOT", self.root), patch.dict(os.environ, {
            "TRACERAG_DATA_DIR": str(absolute)
        }, clear=True):
            self.assertEqual(absolute.resolve(), Settings.from_env().data_dir)
        self.assertEqual(absolute, Settings(data_dir=absolute).data_dir)

    def test_explicit_home_selects_env_and_state_directory(self):
        (self.other / ".env").write_text("TRACERAG_RETRIEVAL=bm25\n", encoding="utf-8")
        with patch("tracerag.config.PROJECT_ROOT", self.root), patch.dict(os.environ, {
            "TRACERAG_HOME": str(self.other)
        }, clear=True):
            settings = Settings.from_env()
        self.assertEqual("bm25", settings.retrieval)
        self.assertEqual((self.other / "data/runtime").resolve(), settings.data_dir)

    @unittest.skipUnless(os.name == "nt", "Windows PowerShell launcher compatibility")
    def test_launcher_parses_in_windows_powershell_51(self):
        script = Path(__file__).resolve().parents[1] / "start.ps1"
        command = (
            "$tokens=$null; $parseErrors=$null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            f"'{str(script).replace(chr(39), chr(39)*2)}',[ref]$tokens,[ref]$parseErrors) | Out-Null; "
            "if($parseErrors.Count){exit 1}"
        )
        # ParseFile reads using the actual PS5.1 file-encoding rules. UTF-8
        # Chinese without BOM previously produced syntax errors on double-click.
        completed = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                                   capture_output=True, timeout=20)
        self.assertEqual(0, completed.returncode, completed.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
