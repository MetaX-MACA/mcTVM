"""Run an AFM batch-128 inference workload on a MACA device via ``torch.export``.

The default input signature and computation follow the inspected Relay graph
in ``afm_bs128.mod``. Parameters are synthetic, not the original model weights:

    main(int32[128, 3] weighted_seq, float32[128, 3, 1] weight,
         int32[128, 1] weighted_seq_seq_length,
         int32[128, 1] sparse_feature_0, int32[128, 1] sparse_feature_1,
         int32[128, 1] sparse_feature_2, int32[128, 3] sequence_sum,
         int32[128, 4] sequence_mean, int32[128, 7] sequence_max)

Run from the repository root after building mcTVM with ``USE_MACA=ON``::

    PYTHONPATH=python LD_LIBRARY_PATH=build/lib \\
        python3 tests/bench/model/batchtest/afm/test_torch_afm_bs128.py

The model uses deterministic random parameters and inputs.  It first verifies
the MACA result against eager PyTorch, then reports end-to-end VM inference
latency (the timing loop excludes compilation and parameter upload).
"""

import argparse
import time

import numpy as np
import torch

import tvm
from torch.export import export
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


class AttentionalFactorizationMachine(torch.nn.Module):
    """The seven-field, four-wide AFM topology in ``afm_bs128.mod``.

    The fields are three sparse IDs plus pooled embeddings for weighted, sum,
    mean, and max sequences.  Their ``C(7, 2) == 21`` pairwise interactions
    form the AFM attention input.
    """

    def __init__(self) -> None:
        super().__init__()
        self.num_fields = 7
        self.embedding_dim = 4
        self.register_buffer("weighted_positions", torch.arange(3, dtype=torch.float32))
        self.sparse_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(6, self.embedding_dim),
                torch.nn.Embedding(1, self.embedding_dim),
                torch.nn.Embedding(2, self.embedding_dim),
            ]
        )
        self.sequence_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(2, self.embedding_dim),
                torch.nn.Embedding(5, self.embedding_dim),
                torch.nn.Embedding(4, self.embedding_dim),
                torch.nn.Embedding(9, self.embedding_dim),
            ]
        )
        self.linear_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(6, 1),
                torch.nn.Embedding(1, 1),
                torch.nn.Embedding(2, 1),
                torch.nn.Embedding(2, 1),
                torch.nn.Embedding(5, 1),
                torch.nn.Embedding(4, 1),
                torch.nn.Embedding(9, 1),
            ]
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1, bias=False),
        )
        self.projection = torch.nn.Linear(self.embedding_dim, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        weighted_seq: torch.Tensor,
        weight: torch.Tensor,
        weighted_seq_seq_length: torch.Tensor,
        sparse_feature_0: torch.Tensor,
        sparse_feature_1: torch.Tensor,
        sparse_feature_2: torch.Tensor,
        sequence_sum: torch.Tensor,
        sequence_mean: torch.Tensor,
        sequence_max: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the nine-input ``afm_bs128.mod`` signature.

        The original graph remaps negative IDs, treats zero as padding for
        unweighted sequences, and applies a sequence-length mask before the
        weighted sequence softmax.  The same rules are expressed here before
        building the seven AFM fields.
        """
        def normalize_index(ids: torch.Tensor, table_size: int) -> torch.Tensor:
            # The benchmark inputs are generated in-range.  Keeping this as
            # an identity also matches the Relay graph, whose take indices
            # are already normalized before lowering and avoids exporting a
            # dead negative-index branch to MACA fast-take lowering.
            return ids

        weighted_ids = normalize_index(weighted_seq, 2)
        weighted_mask = self.weighted_positions < weighted_seq_seq_length.to(torch.float32)
        weighted_mask_float = weighted_mask.unsqueeze(-1).to(torch.float32)
        # Relay uses arithmetic masking, rather than a where/select operation.
        masked_weight = torch.where(weighted_mask.unsqueeze(-1), weight, torch.full_like(weight, -1e9))
        normalized_weight = torch.softmax(masked_weight, dim=1) * weighted_mask_float
        weighted_denominator = weighted_seq_seq_length.to(torch.float32) + 1e-8
        weighted_embedding = torch.sum(
            self.sequence_embeddings[0](weighted_ids) * normalized_weight,
            dim=1, keepdim=True
        ) / weighted_denominator.unsqueeze(-1)

        sum_ids = normalize_index(sequence_sum, 5)
        sum_mask = sequence_sum != 0
        sum_embedding = torch.sum(
            self.sequence_embeddings[1](sum_ids) * sum_mask.unsqueeze(-1), dim=1, keepdim=True
        )

        mean_ids = normalize_index(sequence_mean, 4)
        mean_mask = sequence_mean != 0
        mean_denominator = torch.sum(mean_mask.to(torch.float32), dim=1, keepdim=True) + 1e-8
        mean_embedding = torch.sum(
            self.sequence_embeddings[2](mean_ids) * mean_mask.unsqueeze(-1), dim=1, keepdim=True
        ) / mean_denominator.unsqueeze(-1)

        max_ids = normalize_index(sequence_max, 9)
        max_mask = sequence_max != 0
        max_embedding = torch.max(torch.where(max_mask.unsqueeze(-1), self.sequence_embeddings[3](max_ids), torch.full_like(self.sequence_embeddings[3](max_ids), -1e9)), dim=1, keepdim=True).values
        sparse_embeddings = [
            self.sparse_embeddings[0](normalize_index(sparse_feature_0, 6)),
            self.sparse_embeddings[1](normalize_index(sparse_feature_1, 1)),
            self.sparse_embeddings[2](normalize_index(sparse_feature_2, 2)),
        ]
        embeddings = torch.cat(
            sparse_embeddings + [weighted_embedding, sum_embedding, mean_embedding, max_embedding],
            dim=1,
        )
        # Construct only the 21 upper-triangular field interactions directly.
        # This mirrors the old Relay graph and avoids materializing a 7x7
        # interaction tensor followed by 21 dynamic take operations.
        pairs = torch.cat(
            [embeddings[:, left:left + 1, :] * embeddings[:, right:right + 1, :]
             for left in range(self.num_fields)
             for right in range(left + 1, self.num_fields)],
            dim=1,
        )
        attention = torch.softmax(self.attention(pairs), dim=1)
        pooled = torch.sum(attention * pairs, dim=1)
        afm_logit = self.projection(pooled)

        linear_terms = torch.cat(
            [
                self.linear_embeddings[0](normalize_index(sparse_feature_0, 6)),
                self.linear_embeddings[1](normalize_index(sparse_feature_1, 1)),
                self.linear_embeddings[2](normalize_index(sparse_feature_2, 2)),
                torch.sum(
                    self.linear_embeddings[3](weighted_ids) * normalized_weight,
                    dim=1,
                    keepdim=True,
                ) / weighted_denominator.unsqueeze(-1),
                torch.sum(
                    self.linear_embeddings[4](sum_ids) * sum_mask.unsqueeze(-1),
                    dim=1,
                    keepdim=True,
                ),
                torch.sum(
                    self.linear_embeddings[5](mean_ids) * mean_mask.unsqueeze(-1),
                    dim=1,
                    keepdim=True,
                ) / mean_denominator.unsqueeze(-1),
                torch.max(torch.where(max_mask.unsqueeze(-1), self.linear_embeddings[6](max_ids), torch.full_like(self.linear_embeddings[6](max_ids), -1e9)), dim=1, keepdim=True).values,
            ],
            dim=1,
        )
        linear_logit = torch.sum(linear_terms, dim=1)
        return torch.sigmoid(afm_logit + linear_logit + self.bias)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    return parser.parse_args()


def main(*, benchmark=None, target_config=None) -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError("--warmup must be non-negative and --repeat must be positive")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = AttentionalFactorizationMachine().eval()
    weighted_seq = rng.integers(
        0, 2, size=(args.batch_size, 3), dtype=np.int32
    )
    weight = rng.random((args.batch_size, 3, 1), dtype=np.float32)
    weighted_seq_seq_length = rng.integers(1, 4, size=(args.batch_size, 1), dtype=np.int32)
    sparse_feature_0 = rng.integers(0, 6, size=(args.batch_size, 1), dtype=np.int32)
    sparse_feature_1 = np.zeros((args.batch_size, 1), dtype=np.int32)
    sparse_feature_2 = rng.integers(0, 2, size=(args.batch_size, 1), dtype=np.int32)
    sequence_sum = rng.integers(0, 5, size=(args.batch_size, 3), dtype=np.int32)
    sequence_mean = rng.integers(0, 4, size=(args.batch_size, 4), dtype=np.int32)
    sequence_max = rng.integers(0, 9, size=(args.batch_size, 7), dtype=np.int32)

    exported_program = export(
        model,
        (
            torch.from_numpy(weighted_seq),
            torch.from_numpy(weight),
            torch.from_numpy(weighted_seq_seq_length),
            torch.from_numpy(sparse_feature_0),
            torch.from_numpy(sparse_feature_1),
            torch.from_numpy(sparse_feature_2),
            torch.from_numpy(sequence_sum),
            torch.from_numpy(sequence_mean),
            torch.from_numpy(sequence_max),
        ),
    )
    mod = from_exported_program(exported_program, keep_params_as_input=True)
    mod, params = relax.frontend.detach_params(mod)

    device = tvm.maca(0)
    if not device.exist:
        raise RuntimeError("No MACA device is available at maca:0")
    target = tvm.target.Target({
        "kind": "maca",
        "libs": ["mcdnn", "mcblas", "mccub", "mxexpr"],
        "max_num_threads": 512,
    })
    if target_config is not None:
        target = tvm.target.Target(target_config)
    # Apply target-specific library dispatch and generic GPU schedules.
    with tvm.transform.PassContext(opt_level=3, config={"relax.FuseOps.max_depth": 4096}):
        mod = relax.get_default_pipeline(target)(mod)

    executable = tvm.compile(mod, target=target)

    vm = relax.VirtualMachine(executable, device)
    maca_inputs = (
        tvm.runtime.tensor(weighted_seq, device),
        tvm.runtime.tensor(weight, device),
        tvm.runtime.tensor(weighted_seq_seq_length, device),
        tvm.runtime.tensor(sparse_feature_0, device),
        tvm.runtime.tensor(sparse_feature_1, device),
        tvm.runtime.tensor(sparse_feature_2, device),
        tvm.runtime.tensor(sequence_sum, device),
        tvm.runtime.tensor(sequence_mean, device),
        tvm.runtime.tensor(sequence_max, device),
        *[tvm.runtime.tensor(param, device) for param in params["main"]],
    )
    maca_output = vm["main"](*maca_inputs)[0].numpy()

    with torch.no_grad():
        torch_output = model(
            torch.from_numpy(weighted_seq),
            torch.from_numpy(weight),
            torch.from_numpy(weighted_seq_seq_length),
            torch.from_numpy(sparse_feature_0),
            torch.from_numpy(sparse_feature_1),
            torch.from_numpy(sparse_feature_2),
            torch.from_numpy(sequence_sum),
            torch.from_numpy(sequence_mean),
            torch.from_numpy(sequence_max),
        ).numpy()
    max_abs_diff = float(np.max(np.abs(maca_output - torch_output)))
    np.testing.assert_allclose(maca_output, torch_output, rtol=1e-4, atol=1e-4)
    if benchmark is not None:
        benchmark(vm, maca_inputs, device, target, max_abs_diff)
        return

    for _ in range(args.warmup):
        vm["main"](*maca_inputs)
    device.sync()
    start = time.perf_counter()
    for _ in range(args.repeat):
        vm["main"](*maca_inputs)
    device.sync()
    latency_us = (time.perf_counter() - start) * 1e6 / args.repeat

    print(f"target: {target}")
    print(
        f"input signature: int32[{args.batch_size}, 3], float32[{args.batch_size}, 3, 1], "
        f"4 x int32[{args.batch_size}, 1], int32[{args.batch_size}, 3], "
        f"int32[{args.batch_size}, 4], int32[{args.batch_size}, 7]"
    )
    print(f"output shape: {maca_output.shape}")
    print(f"max absolute difference vs PyTorch: {max_abs_diff:.8g}")
    print(f"average MACA VM inference latency: {latency_us:.2f} us ({args.repeat} runs)")
    print("MACA AFM batch-128 torch.export verification passed.")


if __name__ == "__main__":
    main()
