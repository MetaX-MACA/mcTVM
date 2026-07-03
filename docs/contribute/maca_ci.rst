.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

..   http://www.apache.org/licenses/LICENSE-2.0

.. Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.

Core MACA CI
============

This page documents a local core CI reproduction for this tree.  It keeps only
three lanes:

* ``lint``
* ``cpu``
* ``maca``

The mapping was derived from the Apache TVM community checks observed on the
GitHub change page at ``https://github.com/apache/tvm/pull/19680`` and from
the generated Jenkins files in ``ci/jenkins/generated``.  The local scope is
intentionally smaller than the full community matrix; other matrix checks are
not part of this core script.

Lane mapping
------------

The retained lanes map to the community checks as follows:

* ``lint`` follows the GitHub Actions lint flow:
  ``uv sync --group lint --no-install-project`` followed by
  ``uv run pre-commit run --all-files``.
* ``cpu`` follows the community CPU flow: run
  ``task_config_build_cpu.sh``, build TVM and ``cpptest``, clear pytest XML,
  then run ``task_cpp_unittest.sh`` and ``task_python_unittest.sh`` in two
  unittest shards.
* ``maca`` replaces the community GPU target coverage with MACA.  It builds
  the main MACA configuration, builds a compiler-only MACA configuration, then
  runs the two upstream Python GPU test shards through the MACA wrappers.

Environment requirements
------------------------

The MACA lane requires a Linux host with a visible MACA device and the MACA
SDK installed.  The CMake helper checks ``MACA_PATH`` first and then
``/opt/maca``.  The dynamic loader must be able to find the MACA runtime
libraries:

.. code-block:: bash

   export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/mxshmem/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}

The minimal non-MACA host packages for the retained lanes are:

.. code-block:: bash

   apt-get update
   apt-get install -y \
       python3-pip cmake ninja-build llvm-15-dev libpolly-15-dev libzstd-dev \
       llvm-17-dev libpolly-17-dev libopenblas-dev \
       libgtest-dev libgmock-dev

Install ``uv`` into the Python environment used to run tests:

.. code-block:: bash

   python3 -m pip install --user uv
   export PATH="$HOME/.local/bin:$PATH"

Install the Python packages required by ``pyproject.toml`` and the upstream
test wrappers, or run inside the matching TVM CI image where they are already
provisioned.  The local bare-host validation used:

.. code-block:: bash

   uv pip install --system \
       "pytest~=8.3" "pytest-xdist~=3.6" pytest-rerunfailures==16.1 \
       "pytest-profiling~=1.8" junitparser==4.0.2 \
       "numpy==1.26.*" "scipy~=1.13" "tornado~=6.4" "psutil~=7.0"

The CPU lane uses the upstream CPU CMake config unchanged, including
``USE_TENSORFLOW_PATH=/tensorflow``, ``USE_TFLITE=/opt/tflite``, and
``USE_FLATBUFFERS_PATH=/flatbuffers``.  Those paths are normally provided by
the community CPU image.  Provision equivalent installations if running CPU
parity tests directly on a bare host.

The community CPU image for this tree uses oneDNN 2.x.  The reproduction
script bootstraps oneDNN ``v2.6`` into
``build-maca-ci-deps/onednn-2.6`` and injects that path into the CPU CMake
config by default.  This avoids accidentally using a host ``libdnnl-dev`` from
the oneDNN 3.x API series.

Run commands
------------

Run the retained core lanes sequentially:

.. code-block:: bash

   ./tests/scripts/task_maca_ci.sh

This expands to:

.. code-block:: bash

   ./tests/scripts/task_maca_ci.sh lint cpu maca

Run selected lanes:

.. code-block:: bash

   ./tests/scripts/task_maca_ci.sh lint
   ./tests/scripts/task_maca_ci.sh cpu
   ./tests/scripts/task_maca_ci.sh maca
   ./tests/scripts/task_maca_ci.sh lint cpu

Reuse existing build directories:

.. code-block:: bash

   TVM_MACA_CI_CLEAN_BUILDS=0 ./tests/scripts/task_maca_ci.sh maca

Override build directories when needed:

.. code-block:: bash

   TVM_MACA_CI_CPU_BUILD_DIR=build-maca-ci-cpu \
   TVM_MACA_CI_GPU_BUILD_DIR=build-maca-ci-gpu-maca \
   TVM_MACA_CI_GPU_OTHER_BUILD_DIR=build-maca-ci-gpu-maca-other \
       ./tests/scripts/task_maca_ci.sh cpu maca

Outputs
-------

The default build directories are:

* ``build-maca-ci-cpu``
* ``build-maca-ci-gpu-maca``
* ``build-maca-ci-gpu-maca-other``

Pytest XML output is written to ``build/pytest-results``, matching the
upstream pytest wrapper.

When a lane runs tests, ``task_maca_ci.sh`` sets ``TVM_LIBRARY_PATH`` to the
selected build directory's ``lib`` subdirectory and ``TVM_BUILD_PATH`` to the
selected build directory.  This is required when several TVM builds exist in
one checkout; otherwise Python can load ``build/lib`` from an unrelated
configuration.

The MACA build enables:

* ``USE_MACA=ON``
* ``USE_MCBLAS=ON``
* ``USE_MCDNN=ON``
* ``USE_MCFFT=ON``

This split-library TVM configuration produces ``libtvm_compiler.so`` and
``libtvm_runtime.so``.  Confirm MACA runtime linkage with:

.. code-block:: bash

   ldd build-maca-ci-gpu-maca/lib/libtvm_runtime.so | grep -E 'mcruntime|mcblas|mcdnn|mcfft'
   ldd build-maca-ci-gpu-maca/lib/libtvm_compiler.so | grep -E 'mcruntime|mcblas|mcdnn|mcfft'

MACA wrapper coverage
---------------------

The MACA lane calls the same high-level Python GPU unittest and integration
wrappers as the upstream GPU lane.  The MACA-specific wrappers only restrict
the target selection:

* ``task_python_unittest_maca.sh`` sets ``PYTEST_ADDOPTS="-m gpu"`` and
  ``TVM_TEST_TARGETS=maca``, then runs the upstream
  ``task_python_unittest.sh``.
* ``task_python_unittest_maca.sh`` also mirrors the upstream standalone Vulkan
  codegen smoke by running
  ``tests/python/codegen/test_target_codegen_maca.py`` as a standalone
  ``python-codegen-maca`` smoke.
* ``task_python_integration_maca.sh`` sets ``TVM_TEST_TARGETS=maca``,
  ``TVM_RELAY_TEST_TARGETS=maca``, and ``TVM_INTEGRATION_GPU_ONLY=1``, then
  runs the upstream ``task_python_integration.sh``.  In this TVM revision, the
  upstream integration script performs environment setup and ``tvm-ffi``
  installation only; the MACA wrapper preserves that behavior.

Because the MACA unittest wrapper runs the upstream Python unittest lane,
these files are collected:

* ``tests/python/codegen/test_target_codegen_maca.py``
* ``tests/python/relax/test_codegen_maca.py``

The Relax MACA codegen file contains the dense-add end-to-end mcBLAS test.
That test should pass, not skip, when MACA, mcBLAS, and a MACA device are
available.

Inspect the generated XML after a MACA run:

.. code-block:: bash

   python3 - <<'PY'
   from pathlib import Path
   import xml.etree.ElementTree as ET

   for path in sorted(Path("build/pytest-results").glob("*.xml")):
       root = ET.parse(path).getroot()
       for tc in root.iter("testcase"):
           classname = tc.get("classname") or ""
           name = tc.get("name") or ""
           if "test_target_codegen_maca" in classname or "test_codegen_maca" in classname:
               status = "passed"
               if tc.find("skipped") is not None:
                   status = "skipped"
               if tc.find("failure") is not None:
                   status = "failed"
               if tc.find("error") is not None:
                   status = "error"
               print(path.name, status, classname, name)
   PY

Validation status on the development host
-----------------------------------------

The following checks were run while preparing this script:

* The retained MACA shard flow built the MACA main build and MACA
  compiler-only build, then completed both MACA shards successfully.
* The retained shell scripts passed ``bash -n`` and are executable.
* The CPU lane built with the community CPU config and oneDNN 2.6.  Focused
  reruns confirmed the C++/Python library selection fix and the earlier
  RISC-V/RPC failures were not real regressions.  Full bare-host CPU parity
  still depends on matching the community CPU framework stack and sufficient
  memory for the meta-schedule tests.

On this host, ``ldd`` confirmed that the MACA build links the expected runtime
libraries:

* ``/opt/maca/lib/libmcruntime.so``
* ``/opt/maca/lib/libmcblas.so``
* ``/opt/maca/lib/libmcblasLt.so``
* ``/opt/maca/lib/libmcdnn.so``
* ``/opt/maca/lib/libmcfft.so``

The MACA pytest XML from the validated run recorded the following target
tests as passed:

* ``test_target_codegen_maca.py::test_maca_add``
* ``test_target_codegen_maca.py::test_maca_copy``
* ``test_target_codegen_maca.py::test_maca_vectorized_add``
* ``test_codegen_maca.py::test_mcblas_partition_and_codegen``
* ``test_codegen_maca.py::test_mcblas_matmul_offload``
* ``test_codegen_maca.py::test_mcblas_dense_add_e2e_offload``
* ``test_codegen_maca.py::test_mcdnn_partition_and_codegen``
* ``test_codegen_maca.py::test_mcdnn_softmax_offload``
* ``test_codegen_maca.py::test_mcfft_relax_helpers``
* ``test_codegen_maca.py::test_mcfft_rfft_irfft_offload``

The ``test_mcblas_dense_add_e2e_offload`` result is the dense-add Relax
end-to-end mcBLAS confirmation for this run.

Troubleshooting
---------------

Common bare-host failures and fixes:

* ``Could NOT find zstd`` during LLVM CMake detection: install
  ``libzstd-dev``.
* Missing ``/usr/lib/llvm-15/lib/libPolly.a`` while linking static LLVM:
  install ``libpolly-15-dev``.
* ``uv: command not found`` from Python unittest or integration scripts:
  install ``uv`` into the active Python environment.
* Python imports from the wrong TVM build: use ``task_maca_ci.sh`` or set
  ``TVM_LIBRARY_PATH`` and ``TVM_BUILD_PATH`` to the intended build directory.
* GCC 13 optional/object lifetime warnings are handled in the MACA config with
  ``-Wno-error=maybe-uninitialized`` while keeping the upstream ``-Werror``
  policy for other warnings.
