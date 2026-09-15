"""Run a ComiRec batch-128 inference workload on a MACA device.

The network mirrors ``comirec_bs128.mod``: six user feature embeddings and
the pooled history-genre embedding form a 96-wide user vector.  A two-head
attention network pools a 50-position movie history into two 16-wide interest
vectors; a three-layer user DNN then produces the final interests.  The
candidate movie selects the interest with the largest dot product.

Run from the repository root after building mcTVM with ``USE_MACA=ON``::

    PYTHONPATH=python LD_LIBRARY_PATH=build/lib \\
        python3 tests/bench/model/batchtest/comirec/test_torch_comirec_bs128.py
"""

import argparse
import time

import numpy as np
import torch

import tvm
from torch.export import export
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.target import target


class ComiRec(torch.nn.Module):
    """Two-interest ComiRec model with the ``comirec_bs128.mod`` topology."""

    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.history_size = 50
        self.embedding_dim = 16
        self.num_interests = 2
        self.register_buffer("history_positions", torch.arange(self.history_size, dtype=torch.int32))

        self.user_id_embedding = torch.nn.Embedding(6041, self.embedding_dim)
        self.gender_embedding = torch.nn.Embedding(3, self.embedding_dim)
        self.age_embedding = torch.nn.Embedding(8, self.embedding_dim)
        self.occupation_embedding = torch.nn.Embedding(22, self.embedding_dim)
        self.zip_embedding = torch.nn.Embedding(3440, self.embedding_dim)
        self.hist_genres_embedding = torch.nn.Embedding(19, self.embedding_dim)
        self.hist_movie_id_embedding = torch.nn.Embedding(3707, self.embedding_dim)
        self.position_encoding = torch.nn.Parameter(
            torch.empty(self.history_size, self.embedding_dim)
        )
        torch.nn.init.xavier_uniform_(self.position_encoding)

        self.attention_0 = torch.nn.Linear(self.embedding_dim, 64)
        self.attention_1 = torch.nn.Linear(64, self.num_interests)
        self.user_dnn_0 = torch.nn.Linear(112, 128)
        self.user_dnn_1 = torch.nn.Linear(128, 64)
        self.user_dnn_2 = torch.nn.Linear(64, self.embedding_dim)

    def forward(
        self,
        user_id: torch.Tensor,
        gender: torch.Tensor,
        age: torch.Tensor,
        occupation: torch.Tensor,
        zip_code: torch.Tensor,
        hist_genres: torch.Tensor,
        hist_len: torch.Tensor,
        hist_movie_id: torch.Tensor,
        movie_id: torch.Tensor,
    ) -> torch.Tensor:
        """Return the selected 16-wide user interest for each candidate movie."""
        # ``hist_len`` and the history ID inputs have the [batch, 1] signature
        # recorded by comirec_bs128.mod.  Broadcasting expands them over its
        # fixed 50-position sequence dimension.
        history_mask = self.history_positions < hist_len
        history_mask_f = history_mask.unsqueeze(-1).to(torch.float32)
        hist_len_f = hist_len.to(torch.float32) + 1e-8

        hist_genres_embedding = self.hist_genres_embedding(hist_genres)
        pooled_genres = torch.sum(hist_genres_embedding * history_mask_f, dim=1)
        pooled_genres = pooled_genres / hist_len_f

        user_embedding = torch.cat(
            (
                self.user_id_embedding(user_id),
                self.gender_embedding(gender),
                self.age_embedding(age),
                self.occupation_embedding(occupation),
                self.zip_embedding(zip_code),
                pooled_genres.unsqueeze(1),
            ),
            dim=-1,
        )

        history_embedding = self.hist_movie_id_embedding(hist_movie_id)
        history_embedding = history_embedding + self.position_encoding.unsqueeze(0)
        attention_logits = torch.tanh(self.attention_1(torch.tanh(self.attention_0(history_embedding))))
        masked_logits = torch.where(
            history_mask.unsqueeze(-1), attention_logits, torch.full_like(attention_logits, -1e9)
        )
        attention = torch.softmax(masked_logits, dim=1)
        interests = torch.bmm(attention.transpose(1, 2), history_embedding)

        user_features = user_embedding.repeat(1, self.num_interests, 1)
        interests = torch.cat((user_features, interests), dim=-1)
        interests = torch.relu(self.user_dnn_0(interests))
        interests = torch.relu(self.user_dnn_1(interests))
        interests = self.user_dnn_2(interests)

        candidate_embedding = self.hist_movie_id_embedding(movie_id)
        # comirec_bs128.mod selects the interest by the squared dot product.
        # Squaring is observable when one interest has a negative score.
        interest_scores = torch.sum(interests * candidate_embedding, dim=-1)
        selected_interest = torch.argmax(interest_scores.square(), dim=1, keepdim=True)
        selected_index = selected_interest.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        return torch.gather(interests, dim=1, index=selected_index).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    return parser.parse_args()


def main(*, benchmark=None, target_config=None) -> None:
    args = parse_args()
    if args.batch_size != 128:
        raise ValueError("comirec_bs128.mod has a fixed batch size of 128")
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError("--warmup must be non-negative and --repeat must be positive")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = ComiRec(args.batch_size).eval()
    user_id = rng.integers(0, 6041, size=(args.batch_size, 1), dtype=np.int32)
    gender = rng.integers(0, 3, size=(args.batch_size, 1), dtype=np.int32)
    age = rng.integers(0, 8, size=(args.batch_size, 1), dtype=np.int32)
    occupation = rng.integers(0, 22, size=(args.batch_size, 1), dtype=np.int32)
    zip_code = rng.integers(0, 3440, size=(args.batch_size, 1), dtype=np.int32)
    hist_genres = rng.integers(0, 19, size=(args.batch_size, 1), dtype=np.int32)
    hist_len = rng.integers(1, model.history_size + 1, size=(args.batch_size, 1), dtype=np.int32)
    hist_movie_id = rng.integers(0, 3707, size=(args.batch_size, 1), dtype=np.int32)
    movie_id = rng.integers(0, 3707, size=(args.batch_size, 1), dtype=np.int32)
    torch_inputs = tuple(
        torch.from_numpy(value)
        for value in (
            user_id,
            gender,
            age,
            occupation,
            zip_code,
            hist_genres,
            hist_len,
            hist_movie_id,
            movie_id,
        )
    )

    exported_program = export(model, torch_inputs)
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
    with tvm.transform.PassContext(opt_level=3, config={"relax.FuseOps.max_depth": 4096}):
        mod = relax.get_default_pipeline(target)(mod)
        executable = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(executable, device)

    maca_inputs = tuple(tvm.runtime.tensor(value, device) for value in (*torch_inputs, *params["main"]))
    maca_output = vm["main"](*maca_inputs)[0].numpy()
    with torch.no_grad():
        torch_output = model(*torch_inputs).numpy()
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
    print("input signature: 9 x int32[128, 1] (the last is movie_id)")
    print(f"output shape: {maca_output.shape}")
    print(f"max absolute difference vs PyTorch: {max_abs_diff:.8g}")
    print(f"average MACA VM inference latency: {latency_us:.2f} us ({args.repeat} runs)")
    print("MACA ComiRec batch-128 torch.export verification passed.")


if __name__ == "__main__":
    main()
