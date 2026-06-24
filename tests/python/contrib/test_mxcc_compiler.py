import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tvm.contrib import mxcc


def test_find_mxcc_from_maca_path():
    with TemporaryDirectory() as tmp_dir:
        maca_path = Path(tmp_dir)
        compiler = maca_path / "mxgpu_llvm" / "bin" / "mxcc"
        compiler.parent.mkdir(parents=True)
        compiler.write_text("#!/bin/sh\n", encoding="utf-8")

        with patch.dict(os.environ, {"MACA_PATH": str(maca_path)}, clear=True):
            assert mxcc._find_mxcc() == str(compiler)


def test_find_mxcc_from_default_maca_path():
    default_compiler = "/opt/maca/mxgpu_llvm/bin/mxcc"
    with patch.dict(os.environ, {}, clear=True):
        with patch("tvm.contrib.mxcc.os.path.isfile", return_value=True):
            assert mxcc._find_mxcc() == default_compiler


def test_find_mxcc_from_path():
    with patch.dict(os.environ, {}, clear=True):
        with patch("tvm.contrib.mxcc.os.path.isfile", return_value=False):
            with patch.object(mxcc.shutil, "which", return_value="/usr/bin/mxcc"):
                assert mxcc._find_mxcc() == "/usr/bin/mxcc"
