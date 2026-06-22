import os
from types import SimpleNamespace
from unittest.mock import patch

from tvm.contrib.mxcc import _get_kernels_output_dir


def test_kernel_output_dir_uses_pass_context_first():
    pass_context = SimpleNamespace(config={"maca.kernels_output_dir": "/ctx"})
    with patch.dict(os.environ, {"TVM_MACA_KERNELS_OUTPUT_DIR": "/env"}, clear=True):
        assert _get_kernels_output_dir(pass_context) == "/ctx"


def test_kernel_output_dir_falls_back_to_env():
    pass_context = SimpleNamespace(config={})
    with patch.dict(os.environ, {"TVM_MACA_KERNELS_OUTPUT_DIR": "/env"}, clear=True):
        assert _get_kernels_output_dir(pass_context) == "/env"


def test_kernel_output_dir_accepts_missing_pass_context():
    with patch.dict(os.environ, {"TVM_MACA_KERNELS_OUTPUT_DIR": "/env"}, clear=True):
        assert _get_kernels_output_dir(None) == "/env"


if __name__ == "__main__":
    test_kernel_output_dir_uses_pass_context_first()
    test_kernel_output_dir_falls_back_to_env()
    test_kernel_output_dir_accepts_missing_pass_context()
