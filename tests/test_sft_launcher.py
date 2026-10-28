"""Validate launcher arguments without importing torch or starting training."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(tmp_path):
    capture = tmp_path / "capture.py"
    capture.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake_python = tmp_path / "fake python"
    fake_python.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(capture))} "$@"\n'
    )
    fake_python.chmod(0o755)

    def run(args=(), **overrides):
        env = {
            "PATH": os.environ["PATH"],
            "PYTHON": str(fake_python),
            "TRAIN_JSONL": "prepared_data/train.jsonl",
            **overrides,
        }
        return subprocess.run(
            ["bash", str(ROOT / "audio8_tts_sft.sh"), *args],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )

    return run


def captured_options(result):
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)
    assert argv[:2] == ["-m", "torch.distributed.run"]
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--deepspeed")
    parser.add_argument("--gradient_checkpointing")
    options, _ = parser.parse_known_args(argv)
    return options


def test_defaults_use_bundled_zero2_and_disable_checkpointing(launch):
    options = captured_options(launch())
    config = ROOT / "configs" / "deepspeed_zero2.json"
    assert options.deepspeed == str(config)
    assert json.loads(config.read_text())["zero_optimization"]["stage"] == 2
    assert options.gradient_checkpointing == "false"


@pytest.mark.parametrize("source", ["env", "separate", "equals"])
def test_custom_config_overrides_default(launch, tmp_path, source):
    config = tmp_path / "custom config.json"
    config.write_text('{"zero_optimization": {"stage": 2}}')
    env = {"DEEPSPEED_CONFIG": str(tmp_path / "missing.json")}
    args = []
    if source == "env":
        env["DEEPSPEED_CONFIG"] = str(config)
    elif source == "separate":
        args = ["--deepspeed", str(config)]
    else:
        args = [f"--deepspeed={config}"]
    assert captured_options(launch(args, **env)).deepspeed == str(config)


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--deepspeed", "missing.json"],
        ["--deepspeed"],
        ["--deepspeed="],
        ["--deepspeed", "--bf16", "false"],
    ],
)
def test_bad_config_fails_before_invoking_python(launch, tmp_path, args):
    result = launch(args, DEEPSPEED_CONFIG=str(tmp_path / "missing.json"))
    assert result.returncode == 2
    assert result.stdout == ""
    assert (
        "DeepSpeed config is not a readable file:" in result.stderr
        or "--deepspeed requires a JSON config file path." in result.stderr
    )


def test_checkpointing_can_be_explicitly_enabled(launch):
    options = captured_options(launch(GRADIENT_CHECKPOINTING="true"))
    assert options.gradient_checkpointing == "true"
