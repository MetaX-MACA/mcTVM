import json
import os
from pathlib import Path
from unittest.mock import patch

from tvm.contrib.mxcc import _write_compile_command_log


def test_mxcc_compile_command_log_is_jsonl(tmp_path):
    log_path = tmp_path / "commands.jsonl"

    with patch.dict(os.environ, {"TVM_MACA_COMPILE_COMMAND_LOG": str(log_path)}):
        _write_compile_command_log(
            ["mxcc", "-O3", "-o", "kernel.mcbin", "kernel.maca"],
            "kernel.maca",
            "kernel.mcbin",
        )

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "command": ["mxcc", "-O3", "-o", "kernel.mcbin", "kernel.maca"],
            "source": "kernel.maca",
            "output": "kernel.mcbin",
            "target_format": "mcbin",
        }
    ]


def test_mxcc_compile_command_log_creates_parent_dir(tmp_path):
    log_path = tmp_path / "nested" / "commands.jsonl"

    with patch.dict(os.environ, {"TVM_MACA_COMPILE_COMMAND_LOG": str(log_path)}):
        _write_compile_command_log(["mxcc", "-O2"], "kernel.maca", "kernel.mcbin")

    assert log_path.exists()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_mxcc_compile_command_log_is_jsonl(Path(tmpdir))
        test_mxcc_compile_command_log_creates_parent_dir(Path(tmpdir))
