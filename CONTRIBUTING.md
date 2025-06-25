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

# Contribute Code

You are welcome to contribute to project mcTVM.

If you meet any problem or request a new feature, you're welcome to [create an issue](https://github.com/MetaX-MACA/mcTVM/issues/new/choose).

If you can solve any of [the issues](https://github.com/MetaX-MACA/mcTVM/issues), you're welcome to send the PR to us.

We sincerely appreciate your contribution.  This document explains our workflow and work style.

## Workflow

mcTVM uses this [Git branching model](http://nvie.com/posts/a-successful-git-branching-model/).  The following steps guide usual contributions.

1. Fork

   To contribute, first [create a fork of mcTVM](https://github.com/apache/Metax-MACA/mcTVM/fork)

1. Clone

   To make a copy of your fork to your local computers, please run

   ```bash
   git clone https://github.com/your-github-account/mcTVM
   cd mcTVM
   ```

1. Create the local feature branch

   For daily works like adding a new feature or fixing a bug, please open your feature branch before coding:

   ```bash
   git checkout -b my-cool-stuff
   ```

1. Commit

   commit message format:

   ```bash
   [Tag] Use imperative mood to summarize this commit
   Descriptions
   ```

1. Build and test

1. Keep pulling

   An experienced Git user pulls from the official repo often -- daily or even hourly, so they notice conflicts with others work early, and it's easier to resolve smaller conflicts.

   ```bash
   git remote add upstream https://github.com/Metax-MACA/mcTVM
   git pull upstream 0.18.0
   ```

1. Push and file a pull request

   You can "push" your local work into your forked repo:

   ```bash
   git push origin my-cool-stuff
   ```

   The push allows you to create a pull request, requesting owners of this [official repo](https://github.com/MetaX-MACA/mcTVM) to pull your change into the official one.

   To create a pull request, please follow [these steps](https://help.github.com/articles/creating-a-pull-request/).

   If your change is for fixing an issue, please write ["Fixes <issue-URL>"](https://help.github.com/articles/closing-issues-using-keywords/) in the description section of your pull request.  Github would close the issue when the owners merge your pull request.


1. Delete local and remote branches

   To keep your local workspace and your fork clean, you might want to remove merged branches:

   ```bash
   git push origin :my-cool-stuff
   git checkout 0.18.0
   git pull upstream 0.18.0
   git branch -d my-cool-stuff
   ```

### Code Review

- We regularly review the PR.

- Please answer reviewers' every comment.  If you are to follow the comment, please write "Done"; please give a reason otherwise.

- Reduce the unnecessary commits.  It is recommended to append a sequence of small changes into one commit by running `git commit --amend` instead of `git commit`.

## Coding Standard

### Code Style

Our code styles follows the [TVM C++ Code Styles](https://tvm.apache.org/docs/contribute/code_guide.html#c-code-styles) and [TVM Python Code Styles](https://tvm.apache.org/docs/contribute/code_guide.html#python-code-styles).

Download [Apache-rat](https://creadur.apache.org/rat/download_rat.cgi), and make a copy to `/bin/apache-rat.jar`, install default-jre, clang-format, pre-commit, black，do lint when every commit.

```shell
apt-get install clang-format default-jre
python -m pip install pre-commit black
pre-commit install
```

### Unit Tests

Please remember to add related unit tests, you can follow [TVM Writing Python Tests](https://tvm.apache.org/docs/contribute/code_guide.html#writing-python-tests).
