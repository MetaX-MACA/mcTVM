import ast
import json
import os
from pathlib import Path
from unittest.mock import patch


def _load_mxcc_logging_helper():
    source_path = (
        Path(__file__).resolve().parents[3] / "python" / "tvm" / "contrib" / "mxcc.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or (isinstance(node, ast.FunctionDef) and node.name == "_write_compile_command_log")
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_write_compile_command_log"]


def test_mxcc_compile_command_log_is_jsonl(tmp_path):
    write_log = _load_mxcc_logging_helper()
    log_path = tmp_path / "commands.jsonl"

    with patch.dict(os.environ, {"TVM_MACA_COMPILE_COMMAND_LOG": str(log_path)}):
        write_log(["mxcc", "-O3", "-o", "kernel.mcbin", "kernel.maca"], "kernel.maca", "kernel.mcbin")

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "command": ["mxcc", "-O3", "-o", "kernel.mcbin", "kernel.maca"],
            "source": "kernel.maca",
            "output": "kernel.mcbin",
            "target_format": "mcbin",
        }
    ]


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_mxcc_compile_command_log_is_jsonl(Path(tmpdir))
