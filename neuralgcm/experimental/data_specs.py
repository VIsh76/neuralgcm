# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines observation specifications for NeuralGCM models."""

from __future__ import annotations

import dataclasses
from typing import Generic, TypeVar

import jax
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import typing
import neuralgcm.experimental.jax_datetime as jdt


F = TypeVar('F', cx.Field, cx.Coordinate)


def _extract_coordinate(field: cx.Field):
  """Returns a single coordinate from a fully labeled field."""
  if field.positional_shape:
    raise ValueError(f'Not all dimensions are labeled {field.dims=}')
  return cx.compose_coordinates(*[field.coords[d] for d in field.dims])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TimedObservations(
    Generic[F], typing.ObservationData, typing.ObservationSpecs
):
  """Specifies data or specs for a collection of observations at a given time.

  This object is a self-dual from ObservationData/ObservationSpecs perspective,
  meaning that it can represent either observation specifications or data,
  depending on the type of `self.fields`.

  Attributes:
    fields: A mapping from field names to their values or coordinates.
    timestamp: A timestamp of the observations.
  """

  fields: dict[str, F]
  timestamp: jdt.Datetime

  def get_specs(self) -> TimedObservations:
    if_field = lambda x: isinstance(x, cx.Field)
    to_coord = lambda x: _extract_coordinate(x) if if_field(x) else x
    return jax.tree.map(to_coord, self, is_leaf=if_field)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TimedField(Generic[F], typing.ObservationData, typing.ObservationSpecs):
  """Specifies data or specs for a single observations at a given time.

  This object is a self-dual from ObservationData/ObservationSpecs perspective,
  meaning that it
  can represent either observation specifications or data, depending on the
  type of `self.fields`.

  Attributes:
    field: A field or a coordinate specifying the observation.
    timestamp: A timestamp of the observation.
  """

  field: F
  timestamp: jdt.Datetime | None = None

  def get_specs(self) -> TimedField:
    if_field = lambda x: isinstance(x, cx.Field)
    to_coord = lambda x: _extract_coordinate(x) if if_field(x) else x
    return jax.tree.map(to_coord, self, is_leaf=if_field)
