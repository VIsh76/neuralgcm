from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils

import functools

import gin

import haiku as hk

from neuralgcm import orographies
from neuralgcm import decoders
from neuralgcm import transforms

import numpy as np
import jax.numpy as jnp

from typing import Optional, Dict, Any


@gin.register
class DimensionalPrimitiveToDataStateDecoder(hk.Module):
  """Decoder that converts `StateWithTime` to  `DataState` on the models grid."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      output_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      time_axis: int = 0,
      orography_module: orographies.OrographyModule = orographies.ClippedOrography,
      transform_module: transforms.TransformModule = decoders.DecoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    self.ref_temps = ref_temps[..., np.newaxis, np.newaxis]
    self.coords = coords
    self.physics_specs = physics_specs
    self.velocity_fn = functools.partial(
        spherical_harmonic.vor_div_to_uv_nodal,
        coords.horizontal,
    )
    self.to_nodal_fn = coords.horizontal.to_nodal
    self.redimensionalize_fn = transforms.RedimensionalizeTransform(
        coords,
        dt,
        physics_specs,
        aux_features,
        output_coords=coords,
        inputs_to_units_mapping=inputs_to_units_mapping,
    )

  def __call__(
      self,
      inputs: decoders.ModelState,
      forcing: decoders.Forcing
  ) -> decoders.DataState:
    del forcing

    pe_state = inputs.state
    sim_time = getattr(pe_state, 'sim_time', None)
    u, v = self.velocity_fn(pe_state.vorticity,
                            pe_state.divergence)
    t = self.ref_temps + self.to_nodal_fn(pe_state.temperature_variation)
    tracers = self.to_nodal_fn(pe_state.tracers)
    surface_pressure = jnp.exp(self.to_nodal_fn(pe_state.log_surface_pressure))

    data_state = {
        'sim_time': sim_time,
        'u_component_of_wind': u,
        'v_component_of_wind': v,
        'temperature': t,
        'surface_pressure': surface_pressure,
        **tracers
    }

    return self.redimensionalize_fn(data_state)
