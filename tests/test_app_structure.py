"""
tests/test_app_structure.py
----------------------------
Structural tests for the Streamlit application layer.

These tests deliberately avoid rendering anything (see tests/smoke_pages.py
for headless UI runs) — they verify that the project layout, entry point and
pages are in place, that every Python file compiles, that the language
policy is respected and that the README documents the real setup commands.

Run with:
    pytest tests/test_app_structure.py -v

Covers
------
  TestRequiredFiles   – entry point, six pages, utils trio, docs
  TestPythonCompiles  – every project .py file parses cleanly
  TestPageDiscovery    – page filenames follow the numbered convention
  TestLanguagePolicy   – no loaded/loaded-adjacent word violations
  TestPagesImport      – the shared pages/ package imports cleanly
  TestReadme           – README documents setup and launch commands
"""

import re
from pathlib import Path

import pytest

# Project root (…/paypilot-ai) — two levels above this file.
ROOT = Path(__file__).resolve().parents[1]

# The six feature pages (auto-discovered by Streamlit from pages/).
PAGE_FILES = [
    "pages/01_Dashboard.py",
    "pages/02_Transactions.py",
    "pages/03_Analytics.py",
    "pages/04_Anomalies.py",
    "pages/05_AI_Assistant.py",
    "pages/06_About.py",
]

# Files scanned by the language policy (see TestLanguagePolicy), as
# project-relative POSIX paths.
POLICY_FILES = (
    ["app.py", "README.md"]
    + [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "pages").glob("*.py"))]
    + [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "utils").glob("*.py"))]
)

# Words that must never appear in the UI layer or documentation: the app
# talks about "unusual activity" and "flagged transactions", never about
# wrongdoing.  Word boundaries + case-insensitive.
FORBIDDEN_PATTERN = re.compile(
    r"\b(fraud|fraudulent|criminal|illegal|suspicious)\b", re.IGNORECASE
)

# Every Python file in the project that must compile.
COMPILE_FILES = sorted(
    str(p)
    for folder in ("pages", "utils", "modules", "integrations", "tests")
    for p in (ROOT / folder).glob("*.py")
) + [str(ROOT / "app.py")]


# ---------------------------------------------------------------------------
# Required files exist
# ---------------------------------------------------------------------------


class TestRequiredFiles:
    @pytest.mark.parametrize(
        "relative",
        [
            "app.py",
            "pages/__init__.py",
            "utils/__init__.py",
            "utils/validators.py",
            "utils/formatters.py",
            "utils/logger.py",
            "README.md",
            "requirements.txt",
            ".gitignore",
        ]
        + PAGE_FILES,
    )
    def test_file_exists(self, relative):
        assert (ROOT / relative).is_file(), f"missing: {relative}"

    def test_pages_directory_exists(self):
        assert (ROOT / "pages").is_dir()

    def test_exactly_six_numbered_pages(self):
        numbered = sorted((ROOT / "pages").glob("0*.py"))
        assert len(numbered) == 6
        assert [p.name for p in numbered] == [
            Path(p).name for p in PAGE_FILES
        ]


# ---------------------------------------------------------------------------
# Every Python file compiles (catches syntax errors without importing)
# ---------------------------------------------------------------------------


class TestPythonCompiles:
    @pytest.mark.parametrize("path", COMPILE_FILES, ids=lambda p: Path(p).name)
    def test_compiles(self, path):
        source = Path(path).read_text(encoding="utf-8")
        compile(source, path, "exec")  # raises SyntaxError on failure

    def test_compile_file_list_is_not_empty(self):
        assert len(COMPILE_FILES) >= 15  # 7 UI + 3 utils + 5 modules + tests


# ---------------------------------------------------------------------------
# Page naming convention (Streamlit auto-discovery contract)
# ---------------------------------------------------------------------------


class TestPageDiscovery:
    @pytest.mark.parametrize("page", PAGE_FILES)
    def test_page_has_numeric_prefix(self, page):
        assert re.match(r"^pages/\d{2}_", page)

    def test_app_entry_point_uses_set_page_config(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        assert "st.set_page_config" in source
        assert "PayPilot AI" in source

    def test_every_page_imports_the_shared_layer(self):
        for page in PAGE_FILES:
            source = (ROOT / page).read_text(encoding="utf-8")
            assert "from pages import" in source, f"{page} does not use pages/"

    def test_every_page_has_a_docstring(self):
        for page in PAGE_FILES:
            source = (ROOT / page).read_text(encoding="utf-8")
            assert source.lstrip().startswith('"""'), f"{page} lacks a docstring"


# ---------------------------------------------------------------------------
# Language policy — neutral wording everywhere in the UI layer and docs
# ---------------------------------------------------------------------------


class TestLanguagePolicy:
    @pytest.mark.parametrize("relative", POLICY_FILES, ids=str)
    def test_no_forbidden_words(self, relative):
        text = (ROOT / relative).read_text(encoding="utf-8")
        match = FORBIDDEN_PATTERN.search(text)
        assert match is None, (
            f"{relative} contains '{match.group(0)}' — the UI layer must use "
            "neutral wording ('unusual activity' / 'flagged transaction')."
        )

    def test_policy_scan_covers_all_ui_files(self):
        assert len(POLICY_FILES) >= 11  # app + 7 pages files + 4 utils + README

    def test_ui_mentions_unusual_activity(self):
        """The anomalies page must frame flags as statistical, not moral."""
        source = (ROOT / "pages/04_Anomalies.py").read_text(encoding="utf-8")
        assert "statistically unusual" in source.lower()


# ---------------------------------------------------------------------------
# The shared UI package imports cleanly (sys.path independent)
# ---------------------------------------------------------------------------


class TestPagesImport:
    def test_import_pages_package(self):
        import pages  # noqa: F401

        assert hasattr(pages, "get_transactions")
        assert hasattr(pages, "render_date_filter")
        assert hasattr(pages, "regenerate_dataset")

    def test_utils_importable_from_root(self):
        from utils import formatters, logger, validators  # noqa: F401


# ---------------------------------------------------------------------------
# README documents the real setup and launch commands
# ---------------------------------------------------------------------------


class TestReadme:
    @pytest.fixture(scope="class")
    def readme(self):
        return (ROOT / "README.md").read_text(encoding="utf-8")

    def test_mentions_launch_command(self, readme):
        assert "streamlit run app.py" in readme

    def test_mentions_virtual_environment(self, readme):
        assert "python -m venv" in readme

    def test_mentions_test_commands(self, readme):
        assert "pytest" in readme
        assert "smoke_pages" in readme

    def test_mentions_synthetic_data(self, readme):
        assert "SYNTHETIC" in readme

    def test_mentions_requirements_install(self, readme):
        assert "pip install -r requirements.txt" in readme
