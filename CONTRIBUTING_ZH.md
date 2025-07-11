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

# 贡献代码

欢迎向mcTVM贡献代码.

如果遇到任何问题或有新的需求，请[创建issue](https://github.com/MetaX-MACA/mcTVM/issues/new/choose).

如果解决了任意 [the issues](https://github.com/MetaX-MACA/mcTVM/issues), 欢迎向我们提交PR。

非常感谢您的贡献，下文将介绍我们的工作流和代码风格。

## 工作流

mcTVM 采用 [Git branching model](http://nvie.com/posts/a-successful-git-branching-model/)， 一般步骤如下：

1. 派生

   请先[派生mcTVM](https://github.com/apache/MetaX-MACA/mcTVM/fork)到个人账户下。

1. 克隆

   在本地创建备份：

   ```bash
   git clone https://github.com/your-github-account/mcTVM
   cd mcTVM
   ```

1. 创建分支

   无论是新增特性还是修复问题，在编码前请创建相应分支：

   ```bash
   git checkout -b my-cool-stuff
   ```

1. 提交

   提交信息格式：

   ```bash
   [Tag] Use imperative mood to summarize this commit

   Descriptions
   ```

1. 编译和测试

1. 持续更新

   经验丰富的Git用户习惯频繁地从官方仓库拉取更新--每天甚至每小时，以便及时感知冲突，从而更容易地解决。

   ```bash
   git remote add upstream https://github.com/MetaX-MACA/mcTVM
   git pull upstream 0.18.0
   ```

1. 推送PR

   推送本地分支到私有仓库：

   ```bash
   git push origin my-cool-stuff
   ```

   基于该推送的分支，你可以创建一个拉取请求（Pull Request），请求[仓库](https://github.com/MetaX-MACA/mcTVM)所有者拉取你的修改到官方仓库。

   如何创建拉取请求，请参考[步骤](https://help.github.com/articles/creating-a-pull-request/)。

   如果你的分支是对一个issue的修复，请在拉取请求的描述中输入["Fixes <issue-URL>"](https://help.github.com/articles/closing-issues-using-keywords/)，当你的拉取请求被合入时，Github将关闭该issue。

1. 删除本地和远程分支

   为了保持本地工作区和远程仓库的整洁，你可能希望删除已合入的分支：

   ```bash
   git push origin :my-cool-stuff
   git checkout 0.18.0
   git pull upstream 0.18.0
   git branch -d my-cool-stuff
   ```

### 代码审查

- 我们会规律性的审查所有拉取请求。

- 请处理审查者的每个意见，如果遵守意见请输入“Done”，否则请说明原因。

- 请减少不必要的提交（commit），对于微小修改，建议使用`git commit --amend`合并到前一个提交，而不是采用`git commit`新建一个。

## 编码规范

### 代码风格

参考 [TVM C++ Code Styles](https://tvm.apache.org/docs/contribute/code_guide.html#c-code-styles) 和 [TVM Python Code Styles](https://tvm.apache.org/docs/contribute/code_guide.html#python-code-styles).

下载 [Apache-rat](https://creadur.apache.org/rat/download_rat.cgi), 拷贝到`/bin/apache-rat.jar`, 安装clang-format, pre-commit, black，使在每次commit时进行检查：

```shell
apt-get install clang-format
python -m pip install pre-commit black
pre-commit install
```
### 单元测试

请一定添加相关的单元测试, 参考 [TVM Writing Python Tests](https://tvm.apache.org/docs/contribute/code_guide.html#writing-python-tests).
