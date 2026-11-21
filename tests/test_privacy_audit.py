"""Privacy audit rules. Sensitive-looking strings are assembled at runtime so this file
itself passes the audit."""
import importlib.util
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def audit(repo_root):
    spec = importlib.util.spec_from_file_location("privacy_audit", repo_root / "scripts" / "privacy_audit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["privacy_audit"] = module
    spec.loader.exec_module(module)
    return module


def rules(audit, text, denylist=()):
    return [f.rule for f in audit.scan_text(text, "f.txt", list(denylist))]


AT = "@"
HOME = "/" + "Users" + "/"


@pytest.mark.parametrize("text,rule", [
    ("contact me: jane.doe" + AT + "gmail.com", "email"),
    ("path = '" + HOME + "someone/project'", "home-path"),
    ("cd /" + "home" + "/someone", "home-path"),
    ("C:" + "\\" + "Users" + "\\someone", "home-path"),
    ("key = 'sk-" + "ant-" + "a1B2c3D4e5F6g7H8i9J0'", "anthropic-key"),
    ("OPENAI=sk-" + "Ab3" * 10, "openai-key"),
    ("token " + "ghp_" + "a1" * 20, "github-token"),
    ("aws " + "AKIA" + "ABCDEFGHIJKLMNOP", "aws-key"),
    ("sha " + "deadbeef" * 6, "long-hex-token"),
    ("blob " + "Zm9vYmFy" + "Q1w2E3r4" * 5, "long-random-token"),
])
def test_sensitive_patterns_are_flagged(audit, text, rule):
    assert rule in rules(audit, text)


@pytest.mark.parametrize("text", [
    "author = 1+someone" + AT + "users.noreply.github.com",
    "SEC_USER_AGENT format: Your Name you" + AT + "example.com",
    "Co-Authored-By: Claude <noreply" + AT + "anthropic.com>",
    "def test_a_very_long_but_perfectly_ordinary_function_name_here(): pass",
    "Never hardcode " + HOME + "<name> paths.",
    "MRVL ANET APH MSFT NVDA META BAC SCHW",
])
def test_ordinary_text_is_not_flagged(audit, text):
    assert rules(audit, text) == []


def test_npm_integrity_hashes_are_not_tokens_but_other_tokens_on_the_line_are(audit):
    sri = '"integrity": "sha512-JZnDKK8B0RCDw84FNdDAIpZK+JuJw+s7Lz8nksI7SIuU3UXJJslUthsi+uWBUYOwPFwW7W7PRLRfUKpxjtjFCw==",'
    assert rules(audit, sri) == []
    fake = "aB3dE5fG7hJ9kL1mN3pQ5" + "rS7tU9vW1xY3zA5bC7dE9"   # split so this file passes its own audit
    assert rules(audit, f'{sri} "x": "{fake}"') == ["long-random-token"]


def test_denylist_matches_case_insensitively_without_echoing_value(audit):
    findings = audit.scan_text("Written by JANE EXAMPLE", "notes.md", ["Jane Example"])
    assert [f.rule for f in findings] == ["denylist entry #1"]
    assert "jane" not in str(findings[0]).lower()


def test_allow_marker_skips_a_reviewed_line(audit):
    assert rules(audit, "x" + AT + "gmail.com  # privacy-audit: allow") == []


@pytest.mark.parametrize("path,fragment", [
    (".env", "forbidden file"),
    (".env.local", "forbidden file"),
    (".privacy_denylist", "forbidden file"),
    ("Checkpoint.docx", "forbidden file type"),
    ("data/copilot.db", "forbidden file type"),
    ("data/raw/filing.txt", "local data only"),
    ("demo_outputs/compare.txt", "local data only"),
])
def test_forbidden_paths_are_flagged(audit, path, fragment):
    assert any(fragment in f.rule for f in audit.check_path(path, []))


def test_env_example_is_allowed(audit):
    assert audit.check_path(".env.example", []) == []


def test_env_secret_values_join_the_denylist(audit, tmp_path):
    agent = "Jane Example jane" + AT + "example.com"
    (tmp_path / ".env").write_text(f"SEC_USER_AGENT={agent}\nANTHROPIC_API_KEY=\nOLLAMA_MODEL=qwen3:8b\n")
    (tmp_path / ".privacy_denylist").write_text("# comment\nsecretname\n\n")
    assert audit.load_denylist(tmp_path) == ["secretname", agent]


def test_walk_mode_skips_ignored_folders_and_reports_findings(audit, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "leak.txt").write_text("me" + AT + "gmail.com")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=whatever")
    (tmp_path / "ok.py").write_text("print('hello')\n")
    (tmp_path / "bad.md").write_text("see " + HOME + "someone/notes\n")
    files = audit.list_candidate_files(tmp_path, staged=False)
    assert files == ["bad.md", "ok.py"]
    findings = audit.audit(tmp_path, files, [])
    assert [str(f) for f in findings] == ["bad.md:1: home-path"]


def test_repository_passes_its_own_audit(repo_root):
    result = subprocess.run([sys.executable, str(repo_root / "scripts" / "privacy_audit.py")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
