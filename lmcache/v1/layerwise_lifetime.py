# SPDX-License-Identifier: Apache-2.0
"""Bound remote staging lifetime after a completed layer transfer."""

# Standard
from typing import Any


def release_after_copy(memory_objects: list[Any], stream: Any) -> None:
    """Wait for H2D completion before releasing this layer's staging objects.

    The caller must have submitted the layer transfer on ``stream``. A failed
    synchronization preserves ownership; freeing before completion would allow
    a subsequent remote read to overwrite an in-flight DMA source.
    """
    stream.synchronize()
    for memory_object in memory_objects:
        memory_object.ref_count_down()
        if memory_object.is_pinned:
            memory_object.unpin()
