import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MXCC_PATH = REPO_ROOT / "python" / "tvm" / "contrib" / "mxcc.py"

def _register_global_func(*args, **_kwargs):
    if args and callable(args[0]):
        return args[0]
    return lambda fn: fn


tvm_ffi = types.SimpleNamespace(register_global_func=_register_global_func)
tvm = types.ModuleType("tvm")
tvm.target = types.SimpleNamespace(Target=types.SimpleNamespace(current=lambda: None))
tvm.maca = lambda *_args, **_kwargs: types.SimpleNamespace(exist=False)
tvm.__path__ = []
tvm_contrib = types.ModuleType("tvm.contrib")
tvm_contrib.__path__ = []
tvm_target = types.ModuleType("tvm.target")
tvm_target.Target = object
tvm_base = types.ModuleType("tvm.base")
tvm_base.py_str = lambda value: value.decode("utf-8")
tvm_contrib_utils = types.ModuleType("tvm.contrib.utils")
tvm_contrib_utils.tempdir = lambda: None
sys.modules.setdefault("tvm", tvm)
sys.modules.setdefault("tvm.contrib", tvm_contrib)
sys.modules.setdefault("tvm.target", tvm_target)
sys.modules.setdefault("tvm.base", tvm_base)
sys.modules.setdefault("tvm.contrib.utils", tvm_contrib_utils)

sys.modules.setdefault("tvm_ffi", tvm_ffi)

spec = importlib.util.spec_from_file_location("tvm.contrib.mxcc", MXCC_PATH)
mxcc = importlib.util.module_from_spec(spec)
mxcc.__package__ = "tvm.contrib"
sys.modules["tvm.contrib.mxcc"] = mxcc
assert spec.loader is not None
spec.loader.exec_module(mxcc)


class MxccPathTest(unittest.TestCase):
    def test_maca_path_from_mxcc(self):
        self.assertEqual(
            mxcc._maca_path_from_mxcc("/opt/maca/mxgpu_llvm/bin/mxcc"),
            str(Path("/opt/maca").resolve()),
        )


if __name__ == "__main__":
    unittest.main()
