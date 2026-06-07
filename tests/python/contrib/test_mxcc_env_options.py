import ast
import os
from pathlib import Path
from unittest.mock import patch


def _load_env_options_helper():
    source_path = (
        Path(__file__).resolve().parents[3] / "python" / "tvm" / "contrib" / "mxcc.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and all(alias.name in {"os", "shlex"} for alias in node.names)
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "_get_env_mxcc_options")
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_get_env_mxcc_options"]


def test_env_mxcc_options_are_shell_split():
    get_options = _load_env_options_helper()
    with patch.dict(os.environ, {"TVM_MXCC_OPTIONS": "--flag 'two words'"}, clear=True):
        assert get_options() == ["--flag", "two words"]


def test_env_mxcc_options_default_empty():
    get_options = _load_env_options_helper()
    with patch.dict(os.environ, {}, clear=True):
        assert get_options() == []


if __name__ == "__main__":
    test_env_mxcc_options_are_shell_split()
    test_env_mxcc_options_default_empty()
