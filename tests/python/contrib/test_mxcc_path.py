import unittest
from pathlib import Path
from unittest.mock import patch

from tvm.contrib import mxcc


class MxccPathTest(unittest.TestCase):
    def test_maca_path_from_mxcc(self):
        self.assertEqual(
            mxcc._maca_path_from_mxcc("/opt/maca/mxgpu_llvm/bin/mxcc"),
            str(Path("/opt/maca").resolve()),
        )

    def test_maca_path_from_symlink_target(self):
        with patch("tvm.contrib.mxcc.os.path.realpath") as mock_realpath:
            mock_realpath.side_effect = [
                "/opt/maca/mxgpu_llvm/bin/mxcc",
                str(Path("/opt/maca").resolve()),
            ]
            self.assertEqual(
                mxcc._maca_path_from_mxcc("/usr/bin/mxcc"),
                str(Path("/opt/maca").resolve()),
            )


if __name__ == "__main__":
    unittest.main()
