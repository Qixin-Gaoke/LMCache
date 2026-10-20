# SPDX-License-Identifier: Apache-2.0
"""Bounded NPU gather preserves selected chunk order and output ownership."""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.gpu_connector import utils
from lmcache.v1.multiprocess.transfer_context import base


@pytest.mark.parametrize("extra", [0, 2])
def test_bounded_gather_preserves_outputs(
    monkeypatch: pytest.MonkeyPatch, extra: int
) -> None:
    """Gather selected chunks into caller buffers without touching extras.

    Args:
        monkeypatch: Replace device dispatch and individual native batches.
        extra: Additional output buffers that must remain untouched.
    """
    device = SimpleNamespace(type="npu")
    monkeypatch.setattr(
        utils, "normalize_kv_and_discover_format", lambda *a, **k: ("fmt", [])
    )
    monkeypatch.setattr(utils, "get_device", lambda tensors: device)
    for name in (
        "get_block_size",
        "get_num_layers",
        "get_num_heads",
        "get_head_size",
        "get_num_blocks",
    ):
        monkeypatch.setattr(utils, name, lambda *args: 1)
    monkeypatch.setattr(utils, "make_page_buffer_shape_desc", lambda *a, **k: None)
    monkeypatch.setattr(
        base,
        "resolve_device_ops",
        lambda kind: SimpleNamespace(multi_layer_block_kv_transfer=None),
    )
    monkeypatch.setattr(base, "_detect_block_transfer_accepts_tensor", lambda op: False)
    completed = []
    pending = []

    def gather_batch(*args: object, **kwargs: Any) -> list[list[int]]:
        outputs = kwargs["out"]
        indices = kwargs["chunk_indices"]
        assert len(indices) <= 4
        assert not pending, "previous temporary batch must finish first"
        for output, index in zip(outputs, indices):
            output.append(index)
        pending.extend(indices)
        return outputs[: len(indices)]

    def synchronize(selected: object) -> None:
        assert selected is device
        completed.extend(pending)
        pending.clear()

    gather = base.gather_paged_kv_to_cpu
    monkeypatch.setattr(base, "gather_paged_kv_to_cpu", gather_batch)
    monkeypatch.setattr(base, "synchronize_device", synchronize)
    indices = [7, 2, 5, 1, 8, 0, 6, 3, 4]
    outputs = [[] for _ in range(len(indices) + extra)]
    result = gather({}, list(range(9)), 1, out=outputs, chunk_indices=indices)
    assert result == [[index] for index in indices]
    assert all(result[i] is outputs[i] for i in range(len(indices)))
    assert outputs[len(indices) :] == [[] for _ in range(extra)]
    assert completed == indices and not pending
    if extra == 0:
        assert result is outputs
