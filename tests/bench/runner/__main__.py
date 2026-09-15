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
"""Local Relax runner matching the legacy runner's device profiler metric."""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

import tvm


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def parse_target(value):
    """Translate the legacy CLI syntax to the current Target dictionary API."""
    if value.lstrip().startswith("{"):
        return tvm.target.Target(json.loads(value))
    words = shlex.split(value)
    if not words:
        raise ValueError("Target must not be empty")
    if len(words) == 1:
        return tvm.target.Target(words[0])
    config = {"kind": words[0]}
    options = tvm.target.TargetKind.options_from_name(words[0])
    for word in words[1:]:
        if not word.startswith("-") or "=" not in word:
            raise ValueError(f"Expected target option -name=value, got: {word}")
        key, value = word.lstrip("-").split("=", 1)
        if key in config:
            raise ValueError(f"Duplicate target option: {key}")
        if key not in options:
            raise ValueError(f"Unknown target option: {key}")
        kind = str(options[key])
        if kind in ("IntImm", "int", "Integer", "Bool", "bool"):
            value = (
                int(value)
                if value.lower() not in ("true", "false")
                else int(value.lower() == "true")
            )
        elif "Array" in kind:
            value = value.split(",") if value else []
        elif kind not in ("runtime.String", "String", "ffi.String", "str"):
            raise ValueError(f"Use JSON target syntax for {key} ({kind})")
        config[key] = value
    return tvm.target.Target(config)


def library_support(target):
    """Report pipeline integration, not whether a model actually calls a library."""
    if target.kind.name != "maca":
        return {}
    supported = {
        "mcdnn": "partition available; actual calls depend on matching supported operators",
        "mcblas": "partition available; actual calls depend on matching supported operators",
        "mccub": "not integrated: no MCCUB Relax dispatch/runtime adapter in this repository",
        "mxexpr": (
            "schedule hook exists in legacy MACA injective/broadcast TOPI; "
            "no Relax-specific dispatch"
        ),
    }
    return {
        str(lib): supported.get(str(lib), "unknown: no integration verified")
        for lib in target.attrs.get("libs", [])
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("run-bench")
    bench.add_argument("models", nargs="+", help="Relax benchmark script path or directory")
    bench.add_argument("--profiler", choices=["cupti", "wall"], default="cupti")
    bench.add_argument("--repeats", type=positive, default=10)
    bench.add_argument("--number", type=positive, default=50)
    bench.add_argument("--warmup", type=positive, default=1)
    bench.add_argument(
        "--target",
        default="maca --libs=mcdnn,mcblas,mccub,mxexpr -max_num_threads=512",
        help="TVM target string passed to Relax benchmark scripts",
    )
    bench.add_argument("--libs", default=None, help="Legacy library override (without --target)")
    bench.add_argument(
        "--max-num-threads",
        type=positive,
        default=None,
        help="Legacy thread limit override (without --target)",
    )
    bench.add_argument("--workspace", type=Path, default=Path("tmp/relax_bench"))
    args = parser.parse_args()
    if args.profiler == "cupti" and (
        torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities()
    ):
        parser.error("Device profiler unavailable; cannot substitute wall-clock time")
    args.workspace.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=args.workspace))
    root = Path(__file__).resolve().parents[3]
    try:
        if args.libs is not None or args.max_num_threads is not None:
            if any(arg == "--target" or arg.startswith("--target=") for arg in sys.argv[1:]):
                parser.error("Use either --target or --libs/--max-num-threads")
            target = tvm.target.Target(
                {
                    "kind": "maca",
                    "libs": (
                        args.libs if args.libs is not None else "mcdnn,mcblas,mccub,mxexpr"
                    ).split(",")
                    if args.libs != ""
                    else [],
                    "max_num_threads": args.max_num_threads or 512,
                }
            )
        else:
            target = parse_target(args.target)
    except (ValueError, TypeError) as err:
        parser.error(str(err))
    config = target
    lib_support = library_support(target)
    for lib, status in lib_support.items():
        print(f"Requested library {lib}: {status}", flush=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)

    def discover_models(entries):
        discovered = []
        for entry in entries:
            path = Path(entry)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            path = path.resolve()
            if not path.exists():
                raise ValueError(f"Model input does not exist: {entry}")
            if path.is_dir():
                files = sorted(path.rglob("test_torch_*.py"))
            elif path.suffix == ".py":
                files = [path]
            else:
                raise ValueError(f"Unsupported model input (only Relax .py is supported): {entry}")
            for file in files:
                tree = ast.parse(file.read_text(), filename=str(file))
                main_fn = next(
                    (
                        node
                        for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "main"
                    ),
                    None,
                )
                params = (
                    {arg.arg for arg in main_fn.args.args + main_fn.args.kwonlyargs}
                    if main_fn
                    else set()
                )
                if not {"benchmark", "target_config"} <= params:
                    reason = f"{file}: requires main(*, benchmark=None, target_config=None)"
                    if path.is_dir():
                        print(f"Skipping unsupported script: {reason}", flush=True)
                        continue
                    raise ValueError(reason)
                name = file.stem.removeprefix("test_torch_")
                discovered.append((name, file.resolve()))
        unique = []
        seen = set()
        for item in discovered:
            if item[1] not in seen:
                unique.append(item)
                seen.add(item[1])
        if not unique:
            raise ValueError("No supported Relax benchmark scripts found")
        names = [name for name, _ in unique]
        return [
            (
                name
                if names.count(name) == 1
                else name + "-" + hashlib.sha256(str(path).encode()).hexdigest()[:12],
                path,
            )
            for name, path in unique
        ]

    try:
        models = discover_models(args.models)
    except (ValueError, SyntaxError) as err:
        parser.error(str(err))
    for model, path in models:
        model_output = output / model
        model_output.mkdir(parents=True, exist_ok=True)

        def benchmark(vm, inputs, device, target, error):
            rows = []
            for round_id in range(args.repeats):
                # Match legacy cupti: one warmup outside recording, no L2 flush.
                for _ in range(args.warmup):
                    vm["main"](*inputs)
                if args.profiler == "cupti":
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                    ) as prof:
                        for _ in range(args.number):
                            vm["main"](*inputs)
                    events = prof.key_averages()
                    elapsed = events.total_average().device_time_total / args.number
                    if elapsed <= 0:
                        raise RuntimeError("No device time captured; refusing zero result")
                    details = [
                        dict(
                            name=e.key,
                            calls=e.count,
                            calls_per_run=e.count / args.number,
                            cpu_total_us=e.cpu_time_total,
                            device_total_us=e.device_time_total,
                            device_us_per_run=e.device_time_total / args.number,
                        )
                        for e in events
                    ]
                    write_csv(model_output / f"events-{round_id + 1:02}.csv", details)
                    if round_id == 0:
                        print(events.table(sort_by="device_time_total", row_limit=100))
                    kernels = [
                        e
                        for e in prof.events()
                        if e.device_type == torch.autograd.DeviceType.CUDA
                        and not any(s in e.name.lower() for s in ("memcpy", "memset"))
                    ]
                    launches = len(kernels) / args.number
                    kernel_us = sum(e.time_range.elapsed_us() for e in kernels) / args.number
                    if not kernels:
                        raise RuntimeError("No raw device kernel events captured")
                else:
                    device.sync()
                    start = time.perf_counter()
                    for _ in range(args.number):
                        vm["main"](*inputs)
                    device.sync()
                    elapsed = (time.perf_counter() - start) * 1e6 / args.number
                    launches = kernel_us = None
                rows.append(
                    dict(
                        round=round_id + 1,
                        us_per_run=elapsed,
                        kernel_launches_per_run=launches,
                        kernel_us_per_run=kernel_us,
                    )
                )
                write_csv(model_output / "rounds.csv", rows)
                print(
                    f"{model} round {round_id + 1}/{args.repeats}: {elapsed:.3f} us/run", flush=True
                )
            samples = [r["us_per_run"] for r in rows]
            result = dict(
                model=model,
                model_path=str(path),
                profiler=args.profiler,
                metric="device_time_total/number"
                if args.profiler == "cupti"
                else "synchronized_python_wall_us/number",
                samples_us=samples,
                mean_us=statistics.mean(samples),
                median_us=statistics.median(samples),
                std_us=statistics.pstdev(samples),
                number=args.number,
                repeats=args.repeats,
                warmup=args.warmup,
                clean_l2_cache=False,
                record_shapes=args.profiler == "cupti",
                target=str(target),
                library_support=lib_support,
                max_abs_error=error,
                torch_version=torch.__version__,
                tvm_version=tvm.__version__,
                commit=git.stdout.strip(),
                weights="synthetic, seed=0",
                compile_policy="once, correctness before timing",
                opt_level=3,
                fuse_ops_max_depth=4096,
            )
            (model_output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"{model}: mean={result['mean_us']:.3f}, median={result['median_us']:.3f} us/run")

        spec = importlib.util.spec_from_file_location(f"bench_{model}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_argv = sys.argv
        try:
            sys.argv = [str(path)]
            module.main(benchmark=benchmark, target_config=config)
        finally:
            sys.argv = original_argv
    print(f"Reports: {output.resolve()}")


if __name__ == "__main__":
    main()
