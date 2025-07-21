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
"""Utility for MACA backend"""
import re
import subprocess
import os

import tvm.ffi
import tvm.runtime
import tvm.target

from ..base import py_str
from . import utils


def compile_maca(code, target_format="mcbin", _arch=None, options=None, path_target=None):
    """Compile maca code with MXCC from env.

    Parameters
    ----------
    code : str
        The maca code.

    target_format : str
        The target format of mxcc compiler.

    _arch : str
        The maca architecture.

    options : str or list of str
        The additional options.

    path_target : str, optional
        Output file.

    Return
    ------
    mcbin : bytearray
        The bytearray of fatbin
    """
    temp = utils.tempdir()
    file_name = "tvm_kernels"
    if target_format not in ["mcbin", "mcir", "fatbin"]:
        raise ValueError("target_format must be in mcbin, mcir, fatbin")
    temp_code = temp.relpath(f"{file_name}.maca")
    temp_target = temp.relpath(f"{file_name}.{target_format}")

    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()
    kernels_output_dir = (
        pass_context.config["maca.kernels_output_dir"]
        if "maca.kernels_output_dir" in pass_context.config
        else None
    )
    if kernels_output_dir is not None:
        if not os.path.isdir(kernels_output_dir):
            os.makedirs(kernels_output_dir)
        temp_code = os.path.join(kernels_output_dir, f"{file_name}.maca")
        temp_target = os.path.join(kernels_output_dir, f"{file_name}.{target_format}")

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target
    cmd = ["mxcc"]
    if target_format == "mcbin":
        cmd.append("-device-obj")
    elif target_format == "mcir":
        cmd.extend(["-emit-llvm", "-maca-device-only"])
    else:
        cmd.append("-fatbin")
    cmd.append("-O3")
    if kernels_output_dir is not None:
        cmd += ["-lineinfo"]

    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str of list of str")

    cmd += ["-D__FAST_HALF_CVT__", "-o", file_target]
    cmd += [temp_code]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = code
        msg += "\nCompilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data


def parse_compute_version(compute_version):
    """Parse compute capability string to divide major and minor version

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.0")

    Returns
    -------
    major : int
        major version number
    minor : int
        minor version number
    """
    split_ver = compute_version.split(".")
    try:
        major = int(split_ver[0])
        minor = int(split_ver[1])
        return major, minor
    except (IndexError, ValueError) as err:
        # pylint: disable=raise-missing-from
        raise RuntimeError("Compute version parsing error: " + str(err))


@tvm.ffi.register_func("tvm_callback_maca_have_wmma")
def have_wmma(compute_version=None):
    """Either wmma support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str, optional
        compute capability of a GPU (e.g. "7.0").

    Returns
    -------
    have_wmma : bool
        True if wmma support is provided, False otherwise
    """
    if compute_version is None:
        if tvm.maca(0).exist:
            compute_version = tvm.maca(0).compute_version
        else:
            raise RuntimeError("No MACA runtime found")
    major, _ = parse_compute_version(compute_version)
    # matrix core first introduced in 8.0
    if major >= 8:
        return True

    return False


def have_fp16(compute_version):
    """Either fp16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version: str
        compute capability of a GPU (e.g. "6.0")
    """
    major, _minor = parse_compute_version(compute_version)
    if major >= 10:
        return True

    return False


@tvm.ffi.register_func("tvm_callback_maca_get_arch")
def get_maca_arch(maca_path="/opt/maca"):
    """Utility function to get the MetaX GPU architecture

    Parameters
    ----------
    maca_path : str
        The path to maca installation directory

    Returns
    -------
    gpu_arch : str
        The MetaX GPU architecture
    """
    gpu_arch = "xcore1000"
    # check if maca is installed
    if not os.path.exists(maca_path):
        print("MACA not detected, using default xcore1000")
        return gpu_arch
    try:
        # Execute macainfo command
        macainfo_output = subprocess.check_output([f"{maca_path}/bin/macainfo"]).decode("utf-8")

        # Use regex to match the "Name" field
        match = re.search(r"Name:\s+(XCORE\d+[a-zA-Z]*)", macainfo_output)
        if match:
            gpu_arch = match.group(1)
        return gpu_arch.lower()
    except subprocess.CalledProcessError:
        print(
            f"Unable to execute macainfo command, \
                please ensure MACA is installed and you have an MetaX GPU on your system.\
                    using default {gpu_arch}."
        )
        return gpu_arch


def find_maca_path():
    """Utility function to find MACA path

    Returns
    -------
    path : str
        Path to MACA root.
    """
    if "MACA_PATH" in os.environ:
        return os.environ["MACA_PATH"]
    cmd = ["which", "mxcc"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = out.decode("utf-8").strip()
    if proc.returncode == 0:
        return os.path.realpath(os.path.join(out, "../../.."))
    maca_path = "/opt/maca"
    if os.path.exists(os.path.join(maca_path, "mxgpu_llvm/bin/mxcc")):
        return maca_path
    raise RuntimeError("Cannot find MACA path")


@tvm.ffi.register_func
def tvm_callback_maca_compile(code, target):  # pylint: disable=unused-argument
    """use mxcc to generate fatbin code for better optimization"""
    dev_obj = compile_maca(code, target_format="mcbin")
    return dev_obj


@tvm.ffi.register_func("tvm.contrib.mxcc.get_compute_version")
def get_target_compute_version(target=None):
    """Utility function to get compute capability of compilation target.

    Looks for the target arch in three different places, first in the target input, then the
    Target.current() scope, and finally the GPU device (if it exists).

    Parameters
    ----------
    target : tvm.target.Target, optional
        The compilation target

    Returns
    -------
    compute_version : str
        compute capability of a GPU (e.g. "10.0" of xcore1000)
    """
    # 1. input target object
    # 2. Target.current()
    target = target or tvm.target.Target.current()
    if target and target.mcpu:
        arch = target.mcpu[5:]
        major = arch[:2]
        minor = arch[2:]
        if minor == "00":
            minor = "0"
        return major + "." + minor

    # 3. GPU compute version
    if tvm.maca(0).exist:
        return tvm.maca(0).compute_version

    raise ValueError(
        "No MACA architecture was specified or GPU detected."
        "Try specifying it by adding '-arch=xcorexxxx' to your target."
    )


@tvm.ffi.register_func("tvm.contrib.mxcc.supports_fp8")
def have_fp8(_compute_version):
    """Whether fp8 support is provided in the specified compute capability or not

    Parameters
    ----------
    compute_version : str
        GPU capability
    """
    return False
