# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.graph_load import finish_graph_loads


@pytest.mark.parametrize(
    "mode,expected", [("FULL", 4), ("NONE", 0), ("PIECEWISE", 0)]
)
def test_graph_load_preserves_overlap_except_full_replay(mode, expected):
    calls = []
    adapter = SimpleNamespace(
        num_layers=4, layerwise_retrievers=[object()],
        wait_for_layer_load=lambda name: calls.append(name),
    )
    context = SimpleNamespace(cudagraph_runtime_mode=SimpleNamespace(name=mode))
    finish_graph_loads(adapter, context)
    assert len(calls) == expected
    assert bool(adapter.layerwise_retrievers) == (mode != "FULL")


def test_failed_load_never_reports_drained():
    def fail(name):
        raise RuntimeError("copy failed")
    adapter = SimpleNamespace(
        num_layers=4, layerwise_retrievers=[object()], wait_for_layer_load=fail
    )
    context = SimpleNamespace(cudagraph_runtime_mode=SimpleNamespace(name="FULL"))
    with pytest.raises(RuntimeError, match="copy failed"):
        finish_graph_loads(adapter, context)
    assert adapter.layerwise_retrievers


def test_empty_batch_never_loads():
    adapter = SimpleNamespace(num_layers=4, layerwise_retrievers=[])
    context = SimpleNamespace(cudagraph_runtime_mode=SimpleNamespace(name="FULL"))
    finish_graph_loads(adapter, context)
