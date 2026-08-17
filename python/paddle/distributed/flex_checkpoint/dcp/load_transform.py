# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from paddle import Tensor, dtype


@dataclass(frozen=True)
class LoadTensorMetadata:
    """Metadata for one virtual logical tensor exposed by a load transform."""

    global_shape: tuple[int, ...]
    dtype: str


@runtime_checkable
class LoadTransform(Protocol):
    """Format-independent extension point for checkpoint load transforms.

    A transform exposes virtual logical tensors to AOA, lists the physical
    checkpoint tensors required to materialize each logical tensor, and runs
    only after those physical tensors have been fully assembled.
    """

    def logical_metadata(self) -> dict[str, LoadTensorMetadata]: ...

    def source_keys(self, logical_key: str) -> list[str]: ...

    def apply(
        self,
        logical_key: str,
        source_tensors: dict[str, Tensor],
        output_dtype: dtype,
    ) -> Tensor: ...


def validate_load_transform(
    load_transform: LoadTransform | None,
) -> dict[str, LoadTensorMetadata]:
    if load_transform is None:
        return {}
    if not isinstance(load_transform, LoadTransform):
        raise TypeError(
            "load_transform must provide logical_metadata(), source_keys(), "
            "and apply() methods."
        )

    metadata = load_transform.logical_metadata()
    if not isinstance(metadata, dict):
        raise TypeError("load_transform.logical_metadata() must return a dict.")

    normalized = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                "Load transform logical tensor names must be non-empty strings."
            )
        if not isinstance(value, LoadTensorMetadata):
            raise TypeError(
                f"Metadata for transformed tensor {key!r} must be "
                f"LoadTensorMetadata, got {type(value).__name__}."
            )
        shape = tuple(value.global_shape)
        if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError(
                f"Invalid global shape for transformed tensor {key!r}: {shape}."
            )
        if not isinstance(value.dtype, str) or not value.dtype:
            raise ValueError(
                f"Invalid dtype for transformed tensor {key!r}: "
                f"{value.dtype!r}."
            )
        source_keys = load_transform.source_keys(key)
        if not isinstance(source_keys, (list, tuple)) or not source_keys:
            raise ValueError(
                f"Transformed tensor {key!r} must define at least one source key."
            )
        if any(
            not isinstance(source_key, str) or not source_key
            for source_key in source_keys
        ):
            raise ValueError(
                f"Invalid source key list for transformed tensor {key!r}: "
                f"{source_keys!r}."
            )
        if len(set(source_keys)) != len(source_keys):
            raise ValueError(
                f"Duplicate source keys for transformed tensor {key!r}: "
                f"{source_keys!r}."
            )
        normalized[key] = LoadTensorMetadata(shape, value.dtype)
    return normalized
