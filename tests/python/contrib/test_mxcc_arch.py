# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for MACA architecture detection"""
import os
import subprocess
import unittest
from unittest.mock import patch

from tvm.contrib import mxcc


class TestMacaArchDetection(unittest.TestCase):
    """Test cases for get_maca_arch function"""

    def test_env_maca_arch(self):
        """Test MACA_ARCH environment variable is used when set"""
        with patch.dict(os.environ, {"MACA_ARCH": "xcore2000"}):
            arch = mxcc.get_maca_arch()
            self.assertEqual(arch, "xcore2000")

    def test_env_maca_arch_overrides_maca_path(self):
        """Test MACA_ARCH takes priority even when maca_path is given"""
        with patch.dict(os.environ, {"MACA_ARCH": "xcore3000"}):
            arch = mxcc.get_maca_arch(maca_path="/nonexistent")
            self.assertEqual(arch, "xcore3000")

    def test_maca_not_installed_default(self):
        """Test default arch is returned when MACA is not installed"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                arch = mxcc.get_maca_arch(maca_path="/nonexistent")
                self.assertEqual(arch, "xcore1000")

    def test_maca_path_env_var(self):
        """Test MACA_PATH environment variable is used as fallback path"""
        with patch.dict(os.environ, {"MACA_PATH": "/custom/maca"}, clear=True):
            with patch("os.path.exists", return_value=True):
                with patch(
                    "subprocess.check_output",
                    return_value=b"Name: XCORE1800\n",
                ):
                    arch = mxcc.get_maca_arch()
                    self.assertEqual(arch, "xcore1800")

    def test_oserror_is_caught(self):
        """Test OSError (e.g. missing binary) is caught gracefully"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=True):
                with patch(
                    "subprocess.check_output",
                    side_effect=OSError("macainfo binary not found"),
                ):
                    arch = mxcc.get_maca_arch(maca_path="/valid/path")
                    self.assertEqual(arch, "xcore1000")

    def test_calledprocesserror_is_caught(self):
        """Test CalledProcessError is caught gracefully"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=True):
                with patch(
                    "subprocess.check_output",
                    side_effect=subprocess.CalledProcessError(1, "macainfo"),
                ):
                    arch = mxcc.get_maca_arch(maca_path="/valid/path")
                    self.assertEqual(arch, "xcore1000")

    def test_macainfo_parsing(self):
        """Test parsing of macainfo output"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=True):
                with patch(
                    "subprocess.check_output",
                    return_value=b"Name: XCORE2000\n",
                ):
                    arch = mxcc.get_maca_arch(maca_path="/valid/path")
                    self.assertEqual(arch, "xcore2000")

    def test_macainfo_parsing_no_match(self):
        """Test fallback to default when macainfo output has no matching name"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=True):
                with patch(
                    "subprocess.check_output",
                    return_value=b"Name: UNKNOWN_DEVICE\n",
                ):
                    arch = mxcc.get_maca_arch(maca_path="/valid/path")
                    self.assertEqual(arch, "xcore1000")


if __name__ == "__main__":
    unittest.main()
