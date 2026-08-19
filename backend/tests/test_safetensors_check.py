from __future__ import annotations

import json
import math
import struct
import sys
import types
from pathlib import Path

import pytest

from backend.safetensors_check import (
    MAX_HEADER_BYTES,
    SafeTensorsError,
    validate_safetensors_file,
)


@pytest.fixture(autouse=True)
def _official_safetensors_test_double(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a strict stand-in where the host test Python lacks safetensors.

    Production deliberately fails closed when the official package is absent;
    this test double lets unit tests exercise the subsequent official-parser
    integration path without installing anything into the developer machine.
    """

    item_sizes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E5M2": 1,
        "F8_E4M3": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }

    class FakeSafeOpen:
        def __init__(self, filename: str):
            raw = Path(filename).read_bytes()
            header_size = struct.unpack("<Q", raw[:8])[0]
            self.header = json.loads(raw[8 : 8 + header_size])
            self.tensor_keys = [key for key in self.header if key != "__metadata__"]
            for key in self.tensor_keys:
                item = self.header[key]
                dtype = item["dtype"]
                if dtype not in item_sizes:
                    raise ValueError(f"invalid dtype {dtype}")
                expected = math.prod(item["shape"]) * item_sizes[dtype]
                start, end = item["data_offsets"]
                if end - start != expected:
                    raise ValueError(f"shape byte count mismatch for {key}")

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, exc_type, exc, traceback):  # noqa: ANN001, ANN204
            return False

        def keys(self) -> list[str]:
            return list(self.tensor_keys)

        def metadata(self):  # noqa: ANN201
            return self.header.get("__metadata__")

    module = types.ModuleType("safetensors")
    module.safe_open = lambda filename, **kwargs: FakeSafeOpen(filename)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "safetensors", module)


def _write_safetensors(
    path: Path,
    header: object,
    data: bytes,
    *,
    raw_header: bytes | None = None,
    declared_header_length: int | None = None,
) -> Path:
    encoded = raw_header if raw_header is not None else json.dumps(header, separators=(",", ":")).encode()
    length = len(encoded) if declared_header_length is None else declared_header_length
    path.write_bytes(struct.pack("<Q", length) + encoded + data)
    return path


def test_valid_safetensors_with_metadata_and_contiguous_tensors(tmp_path: Path) -> None:
    path = _write_safetensors(
        tmp_path / "valid.safetensors",
        {
            "__metadata__": {"format": "pt"},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "bias": {"dtype": "F16", "shape": [2], "data_offsets": [4, 8]},
        },
        b"\0" * 8,
    )
    result = validate_safetensors_file(path)
    assert result["size"] == path.stat().st_size
    assert result["tensor_count"] == 2
    assert result["header_bytes"] > 0


@pytest.mark.parametrize(
    ("header", "data", "message"),
    [
        ({}, b"", "no tensors"),
        ({"__metadata__": []}, b"", "metadata"),
        ({"x": []}, b"", "descriptor"),
        ({"x": {"dtype": "", "shape": [], "data_offsets": [0, 0]}}, b"", "dtype"),
        ({"x": {"dtype": "NOT_A_DTYPE", "shape": [1], "data_offsets": [0, 1]}}, b"x", "dtype"),
        ({"x": {"dtype": "F32", "shape": "1", "data_offsets": [0, 4]}}, b"\0" * 4, "shape"),
        ({"x": {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}}, b"\0" * 4, "shape"),
        ({"x": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}, b"\0" * 4, "shape"),
        ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0]}}, b"\0" * 4, "offset"),
        ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [True, 4]}}, b"\0" * 4, "offset"),
        ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}}, b"\0" * 4, "offset"),
        ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [-1, 4]}}, b"\0" * 4, "offset"),
        ({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 5]}}, b"\0" * 4, "offset"),
        # Shape and dtype must agree with the claimed byte range.
        ({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, b"\0" * 4, "shape"),
    ],
)
def test_rejects_invalid_tensor_descriptors(
    tmp_path: Path, header: object, data: bytes, message: str
) -> None:
    path = _write_safetensors(tmp_path / "invalid.safetensors", header, data)
    with pytest.raises(SafeTensorsError, match=message):
        validate_safetensors_file(path)


def test_rejects_overlaps_and_every_kind_of_gap(tmp_path: Path) -> None:
    cases = [
        (
            {
                "a": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
                "b": {"dtype": "U8", "shape": [2], "data_offsets": [2, 4]},
            },
            b"\0" * 4,
            "overlapping|non-contiguous",
        ),
        (
            {"a": {"dtype": "U8", "shape": [3], "data_offsets": [1, 4]}},
            b"\0" * 4,
            "cover|non-contiguous",
        ),
        (
            {"a": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]}},
            b"\0" * 4,
            "cover|non-contiguous",
        ),
        (
            {
                "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
                "b": {"dtype": "U8", "shape": [1], "data_offsets": [2, 3]},
            },
            b"\0" * 3,
            "cover|non-contiguous",
        ),
    ]
    for index, (header, data, message) in enumerate(cases):
        path = _write_safetensors(tmp_path / f"gap-{index}.safetensors", header, data)
        with pytest.raises(SafeTensorsError, match=message):
            validate_safetensors_file(path)


def test_rejects_truncated_or_non_json_headers(tmp_path: Path) -> None:
    too_small = tmp_path / "small.safetensors"
    too_small.write_bytes(b"123456789")
    with pytest.raises(SafeTensorsError, match="too small"):
        validate_safetensors_file(too_small)

    huge = _write_safetensors(
        tmp_path / "huge.safetensors", {}, b"", raw_header=b"{}", declared_header_length=MAX_HEADER_BYTES + 1
    )
    with pytest.raises(SafeTensorsError, match="unreasonable"):
        validate_safetensors_file(huge)

    truncated = _write_safetensors(
        tmp_path / "truncated.safetensors", {}, b"", raw_header=b"{}", declared_header_length=200
    )
    with pytest.raises(SafeTensorsError, match="exceeds"):
        validate_safetensors_file(truncated)

    invalid_json = _write_safetensors(
        tmp_path / "invalid-json.safetensors", {}, b"", raw_header=b"not-json"
    )
    with pytest.raises(SafeTensorsError, match="valid JSON"):
        validate_safetensors_file(invalid_json)

    array_header = _write_safetensors(
        tmp_path / "array.safetensors", {}, b"", raw_header=b"[]"
    )
    with pytest.raises(SafeTensorsError, match="JSON object"):
        validate_safetensors_file(array_header)
