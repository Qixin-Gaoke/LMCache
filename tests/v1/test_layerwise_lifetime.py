# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.layerwise_lifetime import release_after_copy


def test_copy_finishes_before_any_buffer_is_reusable() -> None:
    """Both references and pins survive until transfer completion."""
    calls = []
    stream = SimpleNamespace(synchronize=lambda: calls.append("sync"))
    objects = [SimpleNamespace(ref_count_down=lambda: calls.append("release"),
                               is_pinned=True, unpin=lambda: calls.append("unpin"))]
    release_after_copy(objects, stream)
    assert calls == ["sync", "release", "unpin"]


def test_copy_failure_preserves_staging_ownership() -> None:
    """A failed stream synchronization never returns live DMA buffers."""
    def fail() -> None:
        raise RuntimeError("copy failed")
    objects = [SimpleNamespace(ref_count_down=lambda: pytest.fail("premature release"))]
    with pytest.raises(RuntimeError, match="copy failed"):
        release_after_copy(objects, SimpleNamespace(synchronize=fail))
