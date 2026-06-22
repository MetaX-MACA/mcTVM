import os
from unittest.mock import patch

import pytest

from tvm.contrib.mxcc import _get_mxcc_timeout


def test_mxcc_timeout_accepts_positive_float():
    with patch.dict(os.environ, {"TVM_MXCC_TIMEOUT": "12.5"}, clear=True):
        assert _get_mxcc_timeout() == 12.5


def test_mxcc_timeout_rejects_invalid_value():
    with patch.dict(os.environ, {"TVM_MXCC_TIMEOUT": "0"}, clear=True):
        with pytest.raises(ValueError, match="positive number"):
            _get_mxcc_timeout()


if __name__ == "__main__":
    test_mxcc_timeout_accepts_positive_float()
    test_mxcc_timeout_rejects_invalid_value()
