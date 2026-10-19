# SPDX-License-Identifier: Apache-2.0
"""Complete layerwise loads before graph replay skips Python attention hooks."""

# Standard
from typing import Any


def finish_graph_loads(adapter: Any, forward_context: Any) -> None:
    """Drain pending loads for FULL replay; preserve layer overlap otherwise.

    Args:
        adapter: Connector implementing wait_for_layer_load and layer counters.
        forward_context: vLLM forward context carrying cudagraph_runtime_mode.

    Raises:
        Exception: Propagates failed loads; the caller must not replay the graph.
    """
    mode = getattr(forward_context, "cudagraph_runtime_mode", None)
    if getattr(mode, "name", None) != "FULL" or not adapter.layerwise_retrievers:
        return
    for _ in range(adapter.num_layers):
        adapter.wait_for_layer_load("")
    adapter.layerwise_retrievers = []
