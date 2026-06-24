import os
from unittest.mock import patch
from tvm.contrib.mxcc import _get_env_mxcc_options


def test_env_mxcc_options_are_shell_split():
    with patch.dict(os.environ, {"TVM_MXCC_OPTIONS": "--flag 'two words'"}, clear=True):
        assert _get_env_mxcc_options() == ["--flag", "two words"]


def test_env_mxcc_options_default_empty():
    with patch.dict(os.environ, {}, clear=True):
        assert _get_env_mxcc_options() == []


if __name__ == "__main__":
    test_env_mxcc_options_are_shell_split()
    test_env_mxcc_options_default_empty()
