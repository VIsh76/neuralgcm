# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Basic neural network towers for whirl/gcm codebase.

A tower is a neural network that operates identically over the last two
dimensions, i.e. (longitude, latitude).
"""
from collections import abc
from typing import Callable, Optional, Tuple
from dinosaur import typing
import gin
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm import layers

Array = typing.Array
TowerFactory = typing.TowerFactory
LayerFactory = typing.LayerFactory
MLP = gin.external_configurable(hk.nets.MLP)


@gin.register(denylist=['output_size'])
class ColumnTower(hk.Module):
  """Column tower module parameterized by column_net_factory."""

  def __init__(
      self,
      output_size: int,
      column_net_factory: LayerFactory = gin.REQUIRED,
      checkpoint_tower: bool = False,
      name: Optional[str] = None,
  ):
    """Tower that maps a column_net over two spatial dimensions."""
    super().__init__(name=name)
    column_net = column_net_factory(output_size)
    vmap_last = lambda fn: hk.vmap(fn, in_axes=-1, out_axes=-1, split_rng=False)
    column_tower = vmap_last(vmap_last(column_net))
    if checkpoint_tower:
      column_tower = hk.remat(column_tower)
    self.column_tower = column_tower

  def __call__(self, inputs: Array) -> Array:
    """Applies Column tower to inputs."""
    return self.column_tower(inputs)


@gin.register(denylist=['output_size'])
class ColumnTransformerTower(ColumnTower):
  """Same as ColumnTower, but passes additional transformer inputs."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __call__(
      self,
      inputs: Array,
      latents: Optional[Array] = None,
      positional_encoding: Optional[Array] = None,
  ) -> Array:
    """Applies Column tower to inputs."""
    return self.column_tower(inputs, latents, positional_encoding)


@gin.register(denylist=['output_size'])
class VerticalConvTower(hk.Module):
  """Tower that stacks up layers of Conv1D.

  input shape: [in_channel, level, lon, lat],
  output shape: [output_size, level, lon, lat].
  """

  def __init__(
      self,
      output_size: int,  # The number of channels in the last layer
      channels: abc.Sequence[int] = gin.REQUIRED,
      kernel_shape: int = gin.REQUIRED,
      with_bias: bool = True,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      activate_final: bool = False,
      checkpoint_tower: bool = False,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.activation = activation
    self.output_size = output_size
    self.activate_final = activate_final
    self.checkpoint_tower = checkpoint_tower

    self.layers = []
    channels = list(channels) + [self.output_size]
    for channels_i in channels:
      self.layers.append(layers.ConvLevel(
          output_channels=channels_i,
          kernel_shape=kernel_shape,
          with_bias=with_bias))

  def net(self, inputs: Array) -> Array:
    out = inputs
    num_layers = len(self.layers)
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        out = self.activation(out)
    return out

  def __call__(self, inputs: Array) -> Array:
    vmap_last = lambda fn: hk.vmap(fn, in_axes=-1, out_axes=-1, split_rng=False)
    tower_fn = vmap_last(vmap_last(self.net))
    if self.checkpoint_tower:
      tower_fn = hk.remat(tower_fn)
    return tower_fn(inputs)


@gin.register(denylist=['output_size'])
class Conv2DTower(hk.Module):
  """Two dimensional ConvNet tower module."""

  def __init__(
      self,
      output_size: int,
      num_hidden_units: int = gin.REQUIRED,
      num_hidden_layers: int = gin.REQUIRED,
      kernel_shape: Tuple[int, int] = gin.REQUIRED,
      with_bias: bool = True,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      activate_final: bool = False,
      name: Optional[str] = None,
  ):
    """Tower that stacks up layers of ConvLonLat."""
    super().__init__(name=name)
    self.activation = activation
    self.activate_final = activate_final

    output_sizes = [num_hidden_units] * num_hidden_layers + [output_size]
    self.layers = []
    for output_size in output_sizes:
      self.layers.append(layers.ConvLonLat(
          output_size=output_size,
          kernel_shape=kernel_shape,
          with_bias=with_bias))

  def __call__(self, inputs: Array) -> Array:
    """Applies ConvNet tower to inputs."""
    num_layers = len(self.layers)
    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        out = self.activation(out)
    return out


@gin.register(denylist=['output_size'])
class EpdTower(hk.Module):
  """EPD tower module parameterized by encode/process/decode factories."""

  def __init__(
      self,
      output_size: int,
      latent_size: int = gin.REQUIRED,
      num_process_blocks: int = gin.REQUIRED,
      encode_tower_factory: TowerFactory = gin.REQUIRED,
      process_tower_factory: TowerFactory = gin.REQUIRED,
      decode_tower_factory: TowerFactory = gin.REQUIRED,
      post_encode_activation: Optional[Callable[[Array], Array]] = None,
      pre_decode_activation: Optional[Callable[[Array], Array]] = None,
      final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.output_size = output_size
    self.latent_size = latent_size
    self.num_process_blocks = num_process_blocks
    self.encode_tower_factory = encode_tower_factory
    self.process_tower_factory = process_tower_factory
    self.decode_tower_factory = decode_tower_factory
    self.post_encode_activation = post_encode_activation
    self.pre_decode_activation = pre_decode_activation
    self.final_activation = final_activation

  def __call__(self, inputs: Array) -> Array:
    """Applies EPD tower to inputs."""
    encoded = self.encode_tower_factory(self.latent_size)(inputs)
    if self.post_encode_activation is not None:
      encoded = self.post_encode_activation(encoded)
    current = encoded
    for _ in range(self.num_process_blocks):
      current = current + self.process_tower_factory(self.latent_size)(current)
    if self.pre_decode_activation is not None:
      current = self.pre_decode_activation(current)
    out = self.decode_tower_factory(self.output_size)(current)
    if self.final_activation is not None:
      return self.final_activation(out)
    return out


#### UNET:
class ConvBlockUnet(hk.Module):
    def __init__(self, out_channels: int, activation: Callable, kernel_size:int, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.activation = activation
        self.kernel_size = kernel_size

    def __call__(self, x):
        x = hk.Conv1D(self.out_channels, kernel_shape=self.kernel_size, padding="SAME", data_format='NCW')(x)
        x = self.activation(x)
        x = hk.Conv1D(self.out_channels, kernel_shape=self.kernel_size, padding="SAME", data_format='NCW')(x)
        x = self.activation(x)
        return x
        
class ConvBlockUnet(hk.Module):
    def __init__(self, out_channels: int, activation: Callable, kernel_size:int, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.activation = activation
        self.kernel_size = kernel_size

    def __call__(self, x):
        x = hk.Conv1D(self.out_channels, kernel_shape=self.kernel_size, padding="SAME", data_format='NCW')(x)
        x = self.activation(x)
        x = hk.Conv1D(self.out_channels, kernel_shape=self.kernel_size, padding="SAME", data_format='NCW')(x)
        x = self.activation(x)
        return x


@gin.register(denylist=['output_size'])
class UNet1D2(hk.Module):
    def __init__(self, output_size, num_blocks, activation, kernel_size, levels, latent_size=None, 
                 checkpoint_tower = False,
                 final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
                 name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.num_blocks = num_blocks
        self.activation = activation
        self.kernel_size = kernel_size
        self.latent_size = latent_size
        self.levels = levels
        self.final_activation = final_activation
        self.name = name
        self.checkpoint_tower = checkpoint_tower

        #if self.levels % 2**num_blocks != 0:
        #    raise ValueError(f'levels must be divided by 2**num_bloc, got {self.levels=}, {2**num_blocks=}')
    
    def net(self, column_data: jnp.ndarray, surface_data:jnp.ndarray) -> jnp.ndarray:   
        x = column_data  # (C, L)
        x = x[None, ...]  # add batch: (1, C, L)         

        downs = []
        for i in range(self.num_blocks):
            x = ConvBlockUnet(self.latent_size, self.activation, self.kernel_size)(x)
            downs.append(x)
            x = hk.max_pool(x, window_shape=2, strides=2, padding="SAME", channel_axis=1)

        # Bottleneck: inject surface data
        surface = surface_data[:, :, None]  # (1, 1, F)
        surface_bd = jnp.broadcast_to(surface, (x.shape[0],  surface.shape[1], x.shape[-1])) 

        x = jnp.concatenate([x, surface_bd], axis=1)
        for i in reversed(range(self.num_blocks)):
            x = hk.Conv1DTranspose(self.latent_size, kernel_shape=2, stride=2, padding="SAME", data_format='NCW')(x)
            skip = downs[i]
            # Pad or crop if needed to match shapes
            if skip.shape[-1] != x.shape[-1]:
                min_len = min(skip.shape[-1], x.shape[-1])
                skip = skip[..., :min_len]
                x = x[..., :min_len]
            x = jnp.concatenate([x, skip], axis=1)
            x = ConvBlockUnet(self.latent_size, self.activation, self.kernel_size)(x)

        x = hk.Conv1D(self.output_size, kernel_shape=self.kernel_size, data_format='NCW')(x)
        return x[0]  # remove batch

    def __call__(self, inputs: Array, surface: Array) -> Array:
      # TEST:    
      vmap_last = lambda fn: hk.vmap(fn, in_axes=(-1, -1), out_axes=-1, split_rng=False)
      tower_fn = vmap_last(vmap_last(self.net))
      if self.checkpoint_tower:
        tower_fn = hk.remat(tower_fn)
      output = tower_fn(inputs, surface)
      return output
