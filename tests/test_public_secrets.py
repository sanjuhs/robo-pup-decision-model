"""Synthetic credential shapes must fail current/history scans without leakage."""
from pathlib import Path
import subprocess
import sys

import pytest

SCANNER = Path(__file__).resolve().parents[1] / "scripts/check_public_secrets.py"


@pytest.mark.parametrize("prefix,length", [("rpa_", 48), ("hf_", 34), ("vcp_", 48)])
def test_scanner_reports_presence_without_values_and_checks_deleted_history(tmp_path, prefix, length):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    def scan(*args):
        return subprocess.run(
            [sys.executable, str(SCANNER), *args], cwd=tmp_path, capture_output=True, text=True
        )

    git("init")
    # Repeated synthetic characters are not issued credentials. No complete
    # candidate is stored contiguously in this tracked test fixture.
    secret = prefix + "A" * length
    file = tmp_path / "accidental.txt"
    file.write_text(secret)
    git("add", ".")
    current = scan()
    assert current.returncode == 1
    assert "accidental.txt" in current.stdout
    assert secret not in current.stdout + current.stderr
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    git("rm", "accidental.txt")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "remove")
    assert scan().returncode == 0
    history = scan("--history")
    assert history.returncode == 1
    assert "history blob" in history.stdout
    assert secret not in history.stdout + history.stderr


def test_scanner_redacts_matching_values_in_diagnostic_paths(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    secret = "vcp_" + "A" * 16 + "-" + "B" * 16 + "_" + "C" * 16
    filename = secret + ".txt"
    (tmp_path / filename).write_text(secret)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    result = subprocess.run([sys.executable, str(SCANNER)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "[REDACTED]" in result.stdout
    assert secret not in result.stdout + result.stderr


def test_selected_scanner_allows_placeholders_and_generic_identifiers(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    benign = ["hf_" + "example", "vcp_" + "your_token_here", "A" * 48,
              "eyJ" + "A" * 24 + "." + "B" * 40 + "." + "C" * 40,
              "hf_" + "hub_configuration_cache_settings"]
    (tmp_path / "examples.txt").write_text("\n".join(benign))
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    result = subprocess.run([sys.executable, str(SCANNER)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "0 findings" in result.stdout
