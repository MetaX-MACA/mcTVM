import ast
import os
from pathlib import Path
from unittest.mock import patch


def _load_timeout_helper():
    source_path = (
        Path(__file__).resolve().parents[3] / "python" / "tvm" / "contrib" / "mxcc.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and all(alias.name == "os" for alias in node.names)
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "_get_mxcc_timeout")
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_get_mxcc_timeout"]


def test_mxcc_timeout_accepts_positive_float():
    get_timeout = _load_timeout_helper()
    with patch.dict(os.environ, {"TVM_MXCC_TIMEOUT": "12.5"}, clear=True):
        assert get_timeout() == 12.5


def test_mxcc_timeout_rejects_invalid_value():
    get_timeout = _load_timeout_helper()
    with patch.dict(os.environ, {"TVM_MXCC_TIMEOUT": "0"}, clear=True):
        try:
            get_timeout()
        except ValueError as err:
            assert "positive number" in str(err)
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_mxcc_timeout_accepts_positive_float()
    test_mxcc_timeout_rejects_invalid_value()
