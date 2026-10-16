"""Offline safeguards only: these tests never call Runpod or allocate a GPU."""

import base64
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/runpod_vlm_experiment.py"
SPEC = importlib.util.spec_from_file_location("runpod_vlm_experiment", SCRIPT)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def test_inputs_require_reviewed_committed_files_and_reject_private_changes(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    directory = root / "benchmarks/inputs"
    directory.mkdir(parents=True)
    Image.new("RGB", (32, 24), "red").save(directory / "eye.png")
    (directory / "prompts.json").write_text(json.dumps([
        {"image": "eye.png", "instruction": "fetch the ball", "energy": .9, "held": False},
    ]))
    git(root, "init", "-q")
    git(root, "config", "user.name", "Offline Test")
    git(root, "config", "user.email", "offline@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-qm", "Reviewed public inputs")
    ref = git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(monitor, "ROOT", root)
    packed, count = monitor.pack_inputs(directory)
    assert count == 1
    assert monitor.tracked_input_directory(directory, ref, packed) == "benchmarks/inputs"
    Image.new("RGB", (32, 24), "blue").save(directory / "eye.png")
    changed, _ = monitor.pack_inputs(directory)
    with pytest.raises(ValueError, match="differs"):
        monitor.tracked_input_directory(directory, ref, changed)
    with pytest.raises(ValueError, match="inside"):
        monitor.tracked_input_directory(tmp_path, ref, packed)


def test_untracked_inputs_are_never_uploaded_or_published(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    directory = root / "private-inputs"
    directory.mkdir(parents=True)
    Image.new("RGB", (32, 24), "red").save(directory / "eye.png")
    (directory / "prompts.json").write_text(json.dumps([
        {"image": "eye.png", "instruction": "stop", "energy": .9, "held": False},
    ]))
    git(root, "init", "-q")
    monkeypatch.setattr(monitor, "ROOT", root)
    packed, _ = monitor.pack_inputs(directory)
    with pytest.raises(ValueError, match="committed"):
        monitor.tracked_input_directory(directory, "HEAD", packed)


def test_bootstrap_contains_only_compressed_job_and_stays_below_command_bound():
    script = SCRIPT.with_name("gpu_vlm_job.sh").read_bytes()
    command = monitor.bootstrap_command(script)
    encoded = command.split("printf %s ", 1)[1].split(" ", 1)[0]
    assert gzip.decompress(base64.b64decode(encoded)) == script
    assert len(command.encode()) < 8192
    assert "vlm-input.tar.gz" not in command


def test_failed_allocation_does_not_count_api_time_as_gpu_spend():
    cost = monitor.allocation_costs({"pod_creation_confirmed": False,
                                   "create_requested_at": 1,
                                   "post_error_reconcile_count": 0}, 1.7)
    assert cost["gpu_cost_estimate_usd"] == 0
    assert not cost["gpu_allocation_observed"]
    assert cost["pod_creation_verification_uncertain"]
    assert cost["unconfirmed_gpu_cost_upper_estimate_usd"] == .625
    allocated = monitor.allocation_costs({"pod_creation_confirmed": True,
                                         "confirmed_gpu_rate_usd_hour": 1.19}, 120)
    assert allocated["gpu_cost_estimate_usd"] == pytest.approx(1.19 / 30)
    assert not allocated["pod_creation_verification_uncertain"]


def test_provider_error_classification_retains_only_fixed_flags():
    raw = "Resource unavailable: capacity out of stock. private-marker-value"
    result = monitor.provider_error_categories(raw, 500)
    assert result == {"http_status": 500, "capacity_or_stock": True,
                      "balance_or_credit": False, "payload_size": False,
                      "container_image": False, "validation": False}
    assert "private-marker-value" not in json.dumps(result)
    other = monitor.provider_error_categories(
        "Invalid argument: docker image payload size length exceeds account credit balance funds",
        400,
    )
    assert other["http_status"] == 400 and not other["capacity_or_stock"]
    assert all(other[name] for name in (
        "balance_or_credit", "payload_size", "container_image", "validation",
    ))
