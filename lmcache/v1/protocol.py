# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, Union
import struct

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey, parse_cache_key
from lmcache.v1.memory_management import MemoryFormat

logger = init_logger(__name__)


MAX_KEY_LENGTH = 150
REMOTE_METADATA_FMT: Optional[str] = None
REMOTE_METADATA_BYTES: Optional[int] = None


class ClientCommand(IntEnum):
    PUT = auto()
    GET = auto()
    EXIST = auto()
    LIST = auto()
    HEALTH = auto()


class ServerReturnCode(IntEnum):
    SUCCESS = 200
    FAIL = 400


DTYPE_TO_INT = {
    None: 0,
    torch.half: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
    torch.float: 4,
    torch.float32: 4,
    torch.float64: 5,
    torch.double: 5,
    torch.uint8: 6,
    torch.float8_e4m3fn: 7,
    torch.float8_e5m2: 8,
}

INT_TO_DTYPE = {
    0: None,
    1: torch.half,
    2: torch.float16,
    3: torch.bfloat16,
    4: torch.float,
    5: torch.float64,
    6: torch.uint8,
    7: torch.float8_e4m3fn,
    8: torch.float8_e5m2,
}

# TODO (Jiayi): Add more backends
LOCATION_TO_INT = {
    None: 0,
    "LocalCPUBackend": 1,
    "LocalDiskBackend": 2,
}

INT_TO_LOCATION = {
    0: None,
    1: "LocalCPUBackend",
    2: "LocalDiskBackend",
}


def encode_wire_shape(shape: torch.Size, fmt: MemoryFormat) -> tuple[int, ...]:
    """Encode a layerwise rank-three shape in the fixed four-slot wire header.

    Existing rank-four headers are unchanged. A negative final sentinel marks
    rank three, so empty/dummy rank-four headers remain unambiguous. Both peers
    must support this extension before using a layerwise remote backend.
    """
    if len(shape) == 4 and all(dim >= 0 for dim in shape):
        return tuple(shape)
    if (len(shape) == 3 and fmt in (MemoryFormat.KV_T2D, MemoryFormat.KV_2TD)
            and all(dim > 0 for dim in shape)):
        return (*shape, -1)
    raise ValueError("unsupported remote shape/format")


def decode_wire_shape(shape: tuple[int, ...], fmt: MemoryFormat) -> torch.Size:
    """Restore a validated shape from the fixed-width remote header."""
    if len(shape) != 4:
        raise ValueError("remote header must have four shape slots")
    if shape[-1] == -1:
        if fmt not in (MemoryFormat.KV_T2D, MemoryFormat.KV_2TD) or any(dim <= 0 for dim in shape[:3]):
            raise ValueError("invalid layerwise remote shape")
        return torch.Size(shape[:3])
    if any(dim < 0 for dim in shape):
        raise ValueError("invalid remote shape")
    return torch.Size(shape)


def init_remote_metadata_info(num_groups: int):
    global REMOTE_METADATA_FMT
    global REMOTE_METADATA_BYTES
    # length, fmt, (dtype, shape0, shape1, shape2, shape3) * num_groups
    fmt_length = 2 + 5 * num_groups
    REMOTE_METADATA_FMT = "i" * fmt_length
    REMOTE_METADATA_BYTES = 4 * fmt_length
    logger.info(
        "init remote metadata info with groups: %s, "
        "remote metadata fmt: %s, remote metadata bytes: %s",
        num_groups,
        REMOTE_METADATA_FMT,
        REMOTE_METADATA_BYTES,
    )


def get_remote_metadata_bytes():
    global REMOTE_METADATA_BYTES
    assert REMOTE_METADATA_BYTES is not None
    return REMOTE_METADATA_BYTES


@dataclass
class RemoteMetadata:
    length: int
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]
    fmt: MemoryFormat

    def _prepare_params(self):
        params = [self.length, int(self.fmt.value)]
        for shape, dtype in zip(self.shapes, self.dtypes, strict=True):
            shape = encode_wire_shape(shape, self.fmt)
            params.append(DTYPE_TO_INT[dtype])
            params.append(shape[0])
            params.append(shape[1])
            params.append(shape[2])
            params.append(shape[3])
        return params

    def serialize_into(self, buffer):
        assert REMOTE_METADATA_FMT is not None
        params = self._prepare_params()
        struct.pack_into(REMOTE_METADATA_FMT, buffer, 0, *params)

    def serialize(self) -> bytes:
        assert REMOTE_METADATA_FMT is not None
        params = self._prepare_params()
        packed_bytes = struct.pack(REMOTE_METADATA_FMT, *params)
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes) -> "RemoteMetadata":
        assert REMOTE_METADATA_FMT is not None
        # length, fmt, (dtype, shape0, shape1, shape2, shape3) * num_groups
        result = struct.unpack_from(REMOTE_METADATA_FMT, s)
        length = result[0]
        memory_fmt = MemoryFormat(result[1])
        shapes = []
        dtypes = []
        for i in range(2, len(result), 5):
            shapes.append(decode_wire_shape(tuple(result[i + 1 : i + 5]), memory_fmt))
            dtypes.append(INT_TO_DTYPE[result[i]])

        return RemoteMetadata(
            length,
            shapes,
            dtypes,
            memory_fmt,
        )


# TODO(Jiayi): Server and client message can be merged into one.


@dataclass
class ClientMetaMessage:
    """
    Request message from LMCache workers or servers.
    """

    command: ClientCommand
    key: Union[CacheEngineKey, LayerCacheEngineKey]
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size
    location: Optional[str] = None

    def serialize(self) -> bytes:
        key_str = self.key.to_string()
        assert len(key_str) <= MAX_KEY_LENGTH, (
            f"Key length {len(key_str)} exceeds maximum {MAX_KEY_LENGTH}"
        )

        # NOTE(Jiayi): 4 is the maximum dimension of memory object.
        # Pass in shape [x, 0, 0, 0] if it is a bytes memory object
        wire_shape = encode_wire_shape(self.shape, self.fmt)

        packed_bytes = struct.pack(
            f"iiiiiiiii{MAX_KEY_LENGTH}s",
            self.command.value,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            LOCATION_TO_INT[self.location],
            *wire_shape,
            key_str.encode().ljust(MAX_KEY_LENGTH),
        )
        return packed_bytes

    @staticmethod
    def deserialize(s: bytes) -> "ClientMetaMessage":
        command, length, fmt, dtype, location, shape0, shape1, shape2, shape3, key = (
            struct.unpack(f"iiiiiiiii{MAX_KEY_LENGTH}s", s)
        )
        return ClientMetaMessage(
            ClientCommand(command),
            parse_cache_key(key.decode().strip()),
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            decode_wire_shape((shape0, shape1, shape2, shape3), MemoryFormat(fmt)),
            INT_TO_LOCATION[location],
        )

    @staticmethod
    def packlength() -> int:
        # NOTE: 9 is the number of integers
        return 4 * 9 + MAX_KEY_LENGTH


@dataclass
class ServerMetaMessage:
    """
    Reply message from LMCache workers or servers.
    """

    code: ServerReturnCode
    length: int
    fmt: MemoryFormat
    dtype: Optional[torch.dtype]
    shape: torch.Size
    location: Optional[str] = None

    def serialize(self) -> bytes:
        wire_shape = encode_wire_shape(self.shape, self.fmt)
        packed_bytes = struct.pack(
            "iiiiiiiii",
            self.code.value,
            self.length,
            int(self.fmt.value),
            DTYPE_TO_INT[self.dtype],
            *wire_shape,
            LOCATION_TO_INT[self.location],
        )
        return packed_bytes

    @staticmethod
    def packlength() -> int:
        return 4 * 9

    @staticmethod
    def deserialize(s: bytes) -> "ServerMetaMessage":
        code, length, fmt, dtype, shape0, shape1, shape2, shape3, location = (
            struct.unpack("iiiiiiiii", s)
        )
        return ServerMetaMessage(
            ServerReturnCode(code),
            length,
            MemoryFormat(fmt),
            INT_TO_DTYPE[dtype],
            decode_wire_shape((shape0, shape1, shape2, shape3), MemoryFormat(fmt)),
            INT_TO_LOCATION[location],
        )
