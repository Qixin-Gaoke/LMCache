# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.utils import LayerCacheEngineKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.protocol import (
    ClientCommand, ClientMetaMessage, RemoteMetadata, ServerMetaMessage,
    ServerReturnCode, decode_wire_shape, encode_wire_shape, init_remote_metadata_info,
)


@pytest.mark.parametrize("shape,fmt", [
    ((2, 128, 1024), MemoryFormat.KV_2TD),
    ((128, 2, 1024), MemoryFormat.KV_T2D),
    ((2, 32, 128, 1024), MemoryFormat.KV_2LTD),
    ((0, 0, 0, 0), MemoryFormat.KV_2LTD),
])
def test_remote_shape_roundtrip(shape: tuple, fmt: MemoryFormat) -> None:
    """All three public wire messages preserve dimensions and fixed widths."""
    size = torch.Size(shape)
    key = LayerCacheEngineKey("model", 1, 0, 1234, torch.bfloat16, layer_id=7)
    client = ClientMetaMessage(ClientCommand.PUT, key, 512, fmt, torch.bfloat16, size)
    encoded = client.serialize()
    assert len(encoded) == ClientMetaMessage.packlength()
    assert ClientMetaMessage.deserialize(encoded) == client
    server = ServerMetaMessage(ServerReturnCode.SUCCESS, 512, fmt, torch.bfloat16, size)
    assert len(server.serialize()) == ServerMetaMessage.packlength()
    assert ServerMetaMessage.deserialize(server.serialize()) == server
    init_remote_metadata_info(1)
    remote = RemoteMetadata(512, [size], [torch.bfloat16], fmt)
    assert RemoteMetadata.deserialize(remote.serialize()) == remote
    buffer = bytearray(len(remote.serialize()))
    remote.serialize_into(buffer)
    assert bytes(buffer) == remote.serialize()
    if len(shape) == 4:
        assert encode_wire_shape(size, fmt) == shape


def test_malformed_wire_shape_rejected() -> None:
    """A rank marker cannot silently change an unrelated memory format."""
    with pytest.raises(ValueError):
        encode_wire_shape(torch.Size([2, 128, 1024]), MemoryFormat.KV_2LTD)
    with pytest.raises(ValueError):
        decode_wire_shape((2, 128, 1024, -1), MemoryFormat.KV_2LTD)
    with pytest.raises(ValueError):
        decode_wire_shape((2, -128, 1024, -1), MemoryFormat.KV_2TD)
