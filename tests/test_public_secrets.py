"""A deleted secret must still fail the history check, without leaking its value."""
from pathlib import Path
import subprocess
import sys

SCANNER = Path(__file__).resolve().parents[1] / "scripts/check_public_secrets.py"


def test_scanner_reports_presence_without_values_and_checks_deleted_history(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    def scan(*args):
        return subprocess.run(
            [sys.executable, str(SCANNER), *args], cwd=tmp_path, capture_output=True, text=True
        )

    git("init")
    secret = "rpa_" + "A" * 48  # Synthetic value assembled only inside the test.
    file = tmp_path / "accidental.txt"
    file.write_text(secret)
    git("add", ".")
    current = scan()
    assert current.returncode == 1
    assert "accidental.txt" in current.stdout
    assert secret not in current.stdout + current.stderr
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    git("rm", "accidental.txt")
    assert scan().returncode == 0
    history = scan("--history")
    assert history.returncode == 1
    assert "history blob" in history.stdout
    assert secret not in history.stdout + history.stderr
