<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# MetaX TVM: Open Deep Learning Compiler Stack
English | [中文版](README_ZH.md)

![logo](./imgs/logo.png)

[![GitHub license](https://img.shields.io/github/license/MetaX-MACA/mcTVM?style=flat-square)](./LICENSE)
[![GitHub tag](https://img.shields.io/github/tag/MetaX-MACA/mcTVM?style=flat-square)](https://github.com/MetaX-MACA/mcTVM/releases/?include_prereleases&sort=semver "View GitHub releases")
[![Documentation](https://img.shields.io/badge/documentation-wiki-blue.svg?style=flat-square)](https://tvm.apache.org/docs/)

## Introduction

[Apache TVM](https://github.com/apache/tvm) is one of the earliest and most successful AI compiler，it can take models from popular deep learning frameworks like pytorch and optimize them for diverse hardware. The [TVM community](https://discuss.tvm.apache.org/) is also highly active and [well-documented](https://tvm.apache.org/docs).

This project(mcTVM) is based on [TVM v0.18.0 release](https://github.com/apache/tvm/tree/v0.18.0), and supports [MetaX](https://www.metax-tech.com) GPU.

## Getting Started

### Build From Source

Dependencies:
- MetaX MACA(MetaX Advanced Compute Architecture) programming environment，follow the 《曦云系列_通用计算GPU_快速上手指南》 from [MetaX developer community](https://developer.metax-tech.com)
- Other requirements refer to [TVM Documents](https://tvm.apache.org/docs/install/from_source.html#step-1-install-dependencies)

```shell
git clone https://github.com/MetaX-MACA/mcTVM.git mcTVM
cd mcTVM
git submodule update --init --recursive
rm -rf build && mkdir build && cd build
cp ../cmake/config.cmake ./
# USE_MACA is ON by default
cmake ../ && make -j $(nproc)
export TVM_HOME=/path-to-mcTVM
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

### Using MACA backend

```python
import tvm
# define maca target by kind name
target = tvm.target.Target("maca")
# or specific target by tag name
target = tvm.target.Target("metax/mxc-c500")
# define maca device
dev = tvm.maca()
# or
dev = tvm.device("maca")
```

## Contribute to mcTVM

check out the [CONTRIBUTING.md](./CONTRIBUTING.md)
