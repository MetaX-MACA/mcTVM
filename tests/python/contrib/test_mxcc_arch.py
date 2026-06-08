import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


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

sys.modules.setdefault("tvm_ffi", tvm_ffi)
sys.modules.setdefault("tvm", tvm)
sys.modules.setdefault("tvm.contrib", tvm_contrib)
sys.modules.setdefault("tvm.target", tvm_target)
sys.modules.setdefault("tvm.base", tvm_base)
sys.modules.setdefault("tvm.contrib.utils", tvm_contrib_utils)

spec = importlib.util.spec_from_file_location("tvm.contrib.mxcc", MXCC_PATH)
mxcc = importlib.util.module_from_spec(spec)
mxcc.__package__ = "tvm.contrib"
sys.modules["tvm.contrib.mxcc"] = mxcc
assert spec.loader is not None
spec.loader.exec_module(mxcc)


class MxccArchTest(unittest.TestCase):
    def test_parse_macainfo_arch(self):
        self.assertEqual(mxcc._parse_macainfo_arch("Name: XCORE1000\n"), "XCORE1000")

    def test_get_maca_arch_uses_maca_path_env(self):
        with TemporaryDirectory() as tmp_dir:
            maca_path = Path(tmp_dir)
            (maca_path / "bin").mkdir()
            (maca_path / "bin" / "macainfo").write_text("", encoding="utf-8")
            with patch.dict(os.environ, {"MACA_PATH": str(maca_path)}):
                with patch.object(
                    mxcc.subprocess,
                    "check_output",
                    return_value=b"Name: XCORE2000\n",
                ) as check_output:
                    self.assertEqual(mxcc.get_maca_arch(), "xcore2000")

            check_output.assert_called_once_with([f"{maca_path}/bin/macainfo"])


if __name__ == "__main__":
    unittest.main()
