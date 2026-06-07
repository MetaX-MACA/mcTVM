import ast
from pathlib import Path


def _load_arch_helpers():
    source_path = (
        Path(__file__).resolve().parents[3] / "python" / "tvm" / "contrib" / "mxcc.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"parse_compute_version", "normalize_maca_arch"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def test_normalize_maca_arch_accepts_xcore_and_compute_version():
    helpers = _load_arch_helpers()

    assert helpers["normalize_maca_arch"]("xcore1000") == "10.0"
    assert helpers["normalize_maca_arch"]("xcore1030") == "10.30"
    assert helpers["normalize_maca_arch"]("10.0") == "10.0"


def test_normalize_maca_arch_rejects_unknown_string():
    helpers = _load_arch_helpers()
    try:
        helpers["normalize_maca_arch"]("xcore")
    except RuntimeError as err:
        assert "architecture parsing" in str(err)
    else:
        raise AssertionError("expected RuntimeError")


if __name__ == "__main__":
    test_normalize_maca_arch_accepts_xcore_and_compute_version()
    test_normalize_maca_arch_rejects_unknown_string()
