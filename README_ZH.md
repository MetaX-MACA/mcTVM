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

# Metax TVM: Open Deep Learning Compiler Stack
[English](README.md) | 中文版

![logo](./imgs/logo.png)

[![GitHub license](https://img.shields.io/github/license/MetaX-MACA/mcTVM?style=flat-square)](./LICENSE)
[![GitHub tag](https://img.shields.io/github/tag/MetaX-MACA/mcTVM?style=flat-square)](https://github.com/MetaX-MACA/mcTVM/releases/?include_prereleases&sort=semver "View GitHub releases")
[![Documentation](https://img.shields.io/badge/documentation-wiki-blue.svg?style=flat-square)](https://tvm.apache.org/docs/)

## 简介

[Apache TVM](https://github.com/apache/tvm)是发展最早和最成功的AI编译器之一，支持从主流的深度学习框架比如pytorch导入模型，在优化编译到多种硬件后端，有着非常活跃的[社区讨论](https://discuss.tvm.apache.org/)和完善的[文档支持](https://tvm.apache.org/docs/)。

本项目（mcTVM）在 [TVM v0.18.0 release](https://github.com/apache/tvm/tree/v0.18.0) 的基础上，增加了对[沐曦](https://www.metax-tech.com)（Metax）GPU的支持。

## 快速开始

### 源码编译

依赖：
- 沐曦MACA(Metx Advanced Compute Architecture)编程环境，参见[沐曦开发者社区](https://developer.metax-tech.com)的《曦云系列_通用计算GPU_快速上手指南》
- 其他环境要求请参考[TVM社区文档](https://tvm.apache.org/docs/install/from_source.html#step-1-install-dependencies)

```shell
git clone https://github.com/MetaX-MACA/mcTVM.git mcTVM
cd mcTVM
git submodule update --init --recursive
rm -rf build && mkdir build && cd build
cp ../cmake/config.cmake ./
# USE_MACA 已经默认打开
cmake ../ && make -j $(nproc)
export TVM_HOME=/path-to-mcTVM
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

### 使用MACA后端

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

## 贡献

参见 [CONTRIBUTING_ZH.md](./CONTRIBUTING_ZH.md)
