from unittest.mock import patch

import pytest

from tvm.contrib.mxcc import get_maca_arch, normalize_maca_arch


def test_normalize_maca_arch_accepts_xcore_and_compute_version():
    assert normalize_maca_arch("xcore1000") == "10.0"
    assert normalize_maca_arch("xcore1000b") == "10.0"
    assert normalize_maca_arch("xcore1030") == "10.30"
    assert normalize_maca_arch("10.0") == "10.0"


def test_normalize_maca_arch_rejects_unknown_string():
    with pytest.raises(RuntimeError, match="architecture parsing"):
        normalize_maca_arch("xcore")


def test_get_maca_arch_strips_vendor_suffix():
    with patch("tvm.contrib.mxcc.os.path.exists", return_value=True):
        with patch(
            "tvm.contrib.mxcc.subprocess.check_output",
            return_value=b"Name: XCORE1000B\n",
        ):
            assert get_maca_arch("/opt/maca") == "xcore1000"


if __name__ == "__main__":
    test_normalize_maca_arch_accepts_xcore_and_compute_version()
    test_normalize_maca_arch_rejects_unknown_string()
