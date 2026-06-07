import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_output_dir_helper():
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
        or (isinstance(node, ast.FunctionDef) and node.name == "_get_kernels_output_dir")
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_get_kernels_output_dir"]


def test_kernel_output_dir_uses_pass_context_first():
    get_output_dir = _load_output_dir_helper()
    pass_context = SimpleNamespace(config={"maca.kernels_output_dir": "/ctx"})
    with patch.dict(os.environ, {"TVM_MACA_KERNELS_OUTPUT_DIR": "/env"}, clear=True):
        assert get_output_dir(pass_context) == "/ctx"


def test_kernel_output_dir_falls_back_to_env():
    get_output_dir = _load_output_dir_helper()
    pass_context = SimpleNamespace(config={})
    with patch.dict(os.environ, {"TVM_MACA_KERNELS_OUTPUT_DIR": "/env"}, clear=True):
        assert get_output_dir(pass_context) == "/env"


if __name__ == "__main__":
    test_kernel_output_dir_uses_pass_context_first()
    test_kernel_output_dir_falls_back_to_env()
