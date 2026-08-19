from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


MAX_HEADER_BYTES = 100 * 1024 * 1024
COMMON_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


class SafeTensorsError(ValueError):
    pass


def validate_safetensors_file(path: Path) -> dict[str, Any]:
    """Perform a structural safetensors validation without loading tensor data."""

    path = Path(path)
    file_size = path.stat().st_size
    if file_size < 10:
        raise SafeTensorsError("file is too small to be safetensors")
    with path.open("rb", buffering=0) as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SafeTensorsError("truncated safetensors header length")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length < 2 or header_length > MAX_HEADER_BYTES:
            raise SafeTensorsError("unreasonable safetensors header length")
        if 8 + header_length > file_size:
            raise SafeTensorsError("safetensors header exceeds file size")
        raw_header = stream.read(header_length)
        if len(raw_header) != header_length:
            raise SafeTensorsError("truncated safetensors header")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafeTensorsError("safetensors header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise SafeTensorsError("safetensors header must be a JSON object")
    data_size = file_size - 8 - header_length
    intervals: list[tuple[int, int, str]] = []
    tensor_count = 0
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict):
                raise SafeTensorsError("safetensors metadata must be an object")
            continue
        tensor_count += 1
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            raise SafeTensorsError("invalid tensor descriptor")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if not isinstance(dtype, str) or not dtype or len(dtype) > 64:
            raise SafeTensorsError(f"invalid dtype for tensor {name!r}")
        if (
            not isinstance(shape, list)
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape)
        ):
            raise SafeTensorsError(f"invalid shape for tensor {name!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise SafeTensorsError(f"invalid data offsets for tensor {name!r}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise SafeTensorsError(f"out-of-range data offsets for tensor {name!r}")
        if dtype in COMMON_DTYPE_BYTES:
            element_count = 1
            for dimension in shape:
                element_count *= dimension
            if end - start != element_count * COMMON_DTYPE_BYTES[dtype]:
                raise SafeTensorsError(f"tensor shape does not match byte length for {name!r}")
        intervals.append((start, end, name))
    if tensor_count == 0:
        raise SafeTensorsError("safetensors file contains no tensors")
    intervals.sort()
    previous_end = 0
    for start, end, name in intervals:
        if start < previous_end:
            raise SafeTensorsError(f"overlapping tensor data at {name!r}")
        if start > previous_end:
            raise SafeTensorsError(f"tensor data does not cover the buffer at {name!r}")
        previous_end = end
    # The format requires a compact byte buffer. Enforcing its boundary also
    # catches HTML/error pages with a fabricated short JSON prefix.
    if intervals and (intervals[0][0] != 0 or intervals[-1][1] != data_size):
        raise SafeTensorsError("tensor data does not cover the safetensors data buffer")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SafeTensorsError("the official safetensors parser is required") from exc
    try:
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            parsed_keys = list(tensors.keys())
            tensors.metadata()
    except Exception as exc:
        raise SafeTensorsError(f"official safetensors validation failed: {exc}") from exc
    if len(parsed_keys) != tensor_count:
        raise SafeTensorsError("official safetensors tensor count does not match header")
    return {
        "size": file_size,
        "header_bytes": header_length,
        "tensor_count": tensor_count,
    }
