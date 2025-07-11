#!/usr/bin/env python3
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

import os
import sys
import subprocess
import datetime


def get_changes(base_commit, flag):
    def filter_change(file):
        if file in ["NOTICE", "LICENSE"]:
            return False
        return True

    try:
        changes = subprocess.check_output(
            f"git diff --name-status {base_commit} HEAD | grep '^{flag}' | cut -f2-",
            shell=True,
            text=True,
        )
        changes = list(filter(filter_change, changes.splitlines()))
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while fetching Git changes: {e}")
        changes = []

    return changes


def update_notice_file(added, deleted, modified):
    if not len(added + deleted + modified):
        print("No modifications detected, no need to update the NOTICE file.")
        return

    year = datetime.datetime.now().year
    notice = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../NOTICE")
    content = []

    content.append(
        "The MetaX-MACA/mcTVM project is modified from apache/tvm"
        " (https://github.com/apache/tvm). Please see the LICENSE"
        " for the license for this project.\n"
    )

    if len(modified):
        content.append(
            f"\nThe following files may have been Modified by MetaX Integrated"
            f" Circuits (Shanghai) Co., Ltd. in {year}.\n"
        )
        content.extend(map(lambda x: f"    modified: {x}\n", modified))
        content.append(
            f"Modification copyright {year} MetaX Integrated Circuits (Shanghai)" f" Co., Ltd.\n"
        )

    if len(deleted):
        content.append(
            f"\nThe following files have been deleted by MetaX Integrated Circuits"
            f" (Shanghai) Co., Ltd. in {year}.\n"
        )
        content.extend(map(lambda x: f"    deleted: {x}\n", deleted))

    if len(added):
        content.append(
            f"\nThe following files are newly added by MetaX Integrated Circuits"
            f" (Shanghai) Co., Ltd. in {year}. All rights reserved.\n"
        )
        content.extend(map(lambda x: f"    added: {x}\n", added))

    content.append(
        "\n---------------------------------------------------\n"
        "The following is content of the NOTICE file of apache/tvm:\n"
        "Apache TVM\n"
        "Copyright 2019-2023 The Apache Software Foundation\n"
        "This product includes software developed at\n"
        "The Apache Software Foundation (http://www.apache.org/).\n"
    )

    with open(notice, "w", encoding="utf-8") as f:
        f.writelines(content)
    print(f"update NOTICE file: {notice}")

    try:
        subprocess.run(f"git add {notice}", shell=True, check=True)
        print(f"{notice} has been added to the Git staging area.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while adding {notice} to the Git staging area: {e}")


def main():
    if len(sys.argv) < 2:
        print("ERROR: Please provide the commit ID as a parameter.")
        print("Usage: python3 notice.py <base_commit_id>")

    base_commit = sys.argv[1]

    # check commit ID
    try:
        subprocess.run(
            f"git cat-file -e {base_commit}^0",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        print(f"ERROR: Invalid commit ID: {base_commit}")
        sys.exit(1)

    print(f"Comparsion Baseline: {base_commit} -> HEAD")

    added = get_changes(base_commit, "A")
    deleted = get_changes(base_commit, "D")
    modified = get_changes(base_commit, "M")

    update_notice_file(added, deleted, modified)


if __name__ == "__main__":
    main()
