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

# Relax benchmark runner

Run from the repository root:

```bash
export TVM_LIBRARY_PATH="$PWD/build/lib"
export PYTHONPATH="$PWD/python:$PWD/tests/bench${PYTHONPATH:+:$PYTHONPATH}"

# Recursively discover supported scripts.
python -m runner run-bench tests/bench/model/batchtest \
  --target='maca --libs=mcdnn,mcblas,mccub,mxexpr -max_num_threads=512'

# Run one script.
python -m runner run-bench tests/bench/model/batchtest/comirec/test_torch_comirec_bs128.py \
  --target='maca --libs=mcdnn,mcblas,mccub,mxexpr -max_num_threads=512'

```

The default measurement is CUPTI device time, with 10 rounds, 50 invocations
per round and one warmup invocation before each round. Override these using
`--repeats`, `--number` and `--warmup`. `--profiler=wall` instead measures
synchronized Python wall time; it is a different metric.

The runner translates legacy `-option=value` / `--option=value` target strings
to the current TVM Target dictionary API. JSON target configurations are also
accepted. The complete Target is passed to the script as `target_config`.
For MACA, requested library integration status is printed and recorded in
`summary.json` as `library_support`. MCDNN/MCBlas have pipeline partitions;
actual calls still depend on operator matching. MCCUB has a legacy Relay
TopK strategy/runtime, but no Relax dispatch adapter. `mxexpr` has legacy
injective/broadcast scheduling hooks, but no Relax-specific dispatch. The old
typo `mxepr` is not silently aliased.
Accepting a name in `--libs` does not enable a missing library implementation.
Without `--target`, legacy `--libs` and `--max-num-threads` overrides remain
available; they cannot be combined with an explicit `--target`.

Directory discovery matches `test_torch_*.py` recursively. Scripts must
provide `main(*, benchmark=None, target_config=None)`, use `target_config` when
compiling, verify correctness, then call
`benchmark(vm, inputs, device, target, max_abs_error)`.
AFM and ComiRec currently implement this interface. Other scripts are listed
as skipped during directory discovery; explicitly selecting an unsupported
script is an error. An explicitly selected `.py` file may have any name.
Repeated paths run once. Same-name scripts in different directories receive
distinct report directories. Relay `.mod` files are not supported.

Reports are written under `--workspace` (default `tmp/relax_bench`) in a unique
`run-*/<model>/` directory: `summary.json`, `rounds.csv`, and, for CUPTI,
`events-XX.csv`. Summaries include the resolved model path and actual target.
Model execution failures stop the command with a nonzero exit status.
