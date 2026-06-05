import hashlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MXCC_PATH = REPO_ROOT / "python" / "tvm" / "contrib" / "mxcc.py"

tvm_ffi = types.ModuleType("tvm_ffi")
tvm_ffi.register_global_func = lambda *args, **kwargs: (
    (lambda func: func) if args and callable(args[0]) else (lambda func: func)
)
tvm_ffi.get_global_func = lambda _name: None
sys.modules["tvm_ffi"] = tvm_ffi

tvm = types.ModuleType("tvm")
tvm.maca = lambda *_args, **_kwargs: types.SimpleNamespace(exist=False)
tvm.target = types.ModuleType("tvm.target")
tvm.target.Target = object
sys.modules["tvm"] = tvm
sys.modules["tvm.target"] = tvm.target

tvm_base = types.ModuleType("tvm.base")
tvm_base.py_str = lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value)
sys.modules["tvm.base"] = tvm_base

tvm_contrib = types.ModuleType("tvm.contrib")
tvm_contrib.__path__ = []
sys.modules["tvm.contrib"] = tvm_contrib

tvm_contrib_utils = types.ModuleType("tvm.contrib.utils")
tvm_contrib_utils.tempdir = lambda: None
sys.modules["tvm.contrib.utils"] = tvm_contrib_utils

spec = importlib.util.spec_from_file_location("tvm.contrib.mxcc", MXCC_PATH)
mxcc = importlib.util.module_from_spec(spec)
mxcc.__package__ = "tvm.contrib"
sys.modules["tvm.contrib.mxcc"] = mxcc
assert spec.loader is not None
spec.loader.exec_module(mxcc)


class TestMXCCKernelDumpName(unittest.TestCase):

    def test_kernel_file_name_is_stable_and_content_addressed(self):
        source = "extern \"C\" __global__ void add(float* x) { x[0] += 1.0f; }"
        expected_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

        self.assertEqual(mxcc._kernel_file_name(source), f"tvm_kernels_{expected_hash}")
        self.assertEqual(mxcc._kernel_file_name(source), mxcc._kernel_file_name(source))

    def test_kernel_file_name_changes_with_source(self):
        self.assertNotEqual(
            mxcc._kernel_file_name("extern \"C\" __global__ void a() {}"),
            mxcc._kernel_file_name("extern \"C\" __global__ void b() {}"),
        )


if __name__ == "__main__":
    unittest.main()
