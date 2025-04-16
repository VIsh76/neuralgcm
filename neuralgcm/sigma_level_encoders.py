
from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import pytree_utils
from dinosaur import spherical_harmonic
from dinosaur import vertical_interpolation
from dinosaur import xarray_utils

import functools

import gin

import haiku as hk

import jax
import jax.numpy as jnp

from neuralgcm import orographies
from neuralgcm import transforms
from neuralgcm import encoders

import numpy as np

from typing import Any, Dict, Optional


class _DimensionalDataStateToPrimitiveEncoder(hk.Module):
  """Encoder that converts `DataState` on the models grid to `StateWithTime`."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      time_axis: int = 0,
      orography_module: orographies.OrographyModule = orographies.ClippedOrography,
      transform_module: transforms.TransformModule = encoders.EncoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    self.nondimensionalize_fn = transforms.NondimensionalizeTransform(
        coords,
        dt,
        physics_specs,
        aux_features,
        coords,
        inputs_to_units_mapping=inputs_to_units_mapping,
    )

    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    self.ref_temps = ref_temps[..., np.newaxis, np.newaxis]
    self.coords = coords
    self.slice_fn = functools.partial(
        pytree_utils.slice_along_axis, axis=time_axis, idx=-1)
    self.curl_and_div_fn = functools.partial(
        spherical_harmonic.uv_nodal_to_vor_div_modal,
        coords.horizontal,
    )

  def __call__(
      self,
      inputs: encoders.DataState,
      forcing: encoders.Forcing,
  ) -> primitive_equations.StateWithTime:
    del forcing

    data_state = self.nondimensionalize_fn(inputs)
    data_state = self.slice_fn(data_state) # TODO: <--- do we need this???
    data_state = coordinate_systems.maybe_to_nodal(data_state, self.coords)
    

    #### My contribution (Regridding):
    
    u, v = self.coords.physics_to_dycore_sharding(
        (data_state['u_component_of_wind'], data_state['v_component_of_wind'])
    )
    vorticity, divergence = self.coords.dycore_to_physics_sharding(
        self.curl_and_div_fn(u, v)
    )
    pe_state = primitive_equations.StateWithTime(
        divergence=divergence,
        vorticity=vorticity,
        temperature_variation=(data_state['temperature'] - self.ref_temps),
        log_surface_pressure=jnp.log(data_state['surface_pressure']),
        sim_time=data_state['sim_time'],
        tracers={k: data_state[k] for k in ('specific_humidity','specific_cloud_ice_water_content','specific_cloud_liquid_water_content')},        
    )

    pe_state = coordinate_systems.maybe_to_modal(pe_state, self.coords)
    return encoders.ModelState(state=pe_state)


@gin.register
class _DimensionalDataStateToPrimitiveEncoderWithMemory(hk.Module):
  """Encoder that converts `DataState` on the models grid to `StateWithTime`."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      time_axis: int = 0,
      orography_module: orographies.OrographyModule = orographies.ClippedOrography,
      transform_module: transforms.TransformModule = encoders.EncoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.encoder_fn = DimensionalDataStateToPrimitiveEncoder(
        coords=coords,
        dt=dt,
        physics_specs=physics_specs,
        aux_features=aux_features,
        input_coords=coords,
        inputs_to_units_mapping=inputs_to_units_mapping,
        time_axis=time_axis,
        orography_module=orography_module,
        transform_module=transform_module,
        name=name,
    )

  def __call__(
      self,
      inputs: encoders.DataState,
      forcing: encoders.Forcing,
  ) -> primitive_equations.StateWithTime:

    memory = self.encoder_fn(inputs, forcing=forcing)
    model_state = self.encoder_fn(inputs, forcing=forcing)
    return encoders.ModelState(
        state=model_state.state,
        memory=memory.state,
    )

# %% My Code:
from dinosaur import weatherbench_utils
from dinosaur import typing


class DimensionalDataStateToPrimitiveEncoder(hk.Module):
  """Encoder that converts `DataState` on the models grid to `StateWithTime`."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      time_axis: int = 0,
      orography_module: orographies.OrographyModule = orographies.ClippedOrography,
      transform_module: transforms.TransformModule = encoders.EncoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    self.nondimensionalize_fn = transforms.NondimensionalizeTransform(
        coords,
        dt,
        physics_specs,
        aux_features,
        coords,
        inputs_to_units_mapping=inputs_to_units_mapping,
    )

    ref_temps = aux_features[xarray_utils.REF_TEMP_KEY]
    self.ref_temps = ref_temps[..., np.newaxis, np.newaxis]
    self.coords = coords
    self.slice_fn = functools.partial(
        pytree_utils.slice_along_axis, axis=time_axis, idx=-1)
    self.curl_and_div_fn = functools.partial(
        spherical_harmonic.uv_nodal_to_vor_div_modal,
        coords.horizontal,
    )
    modal_orography_init_fn = orography_module(coords, dt, physics_specs, aux_features)
    modal_orography = modal_orography_init_fn()  # pytype: disable=not-callable  # jax-ndarray
    self.slice_fn = functools.partial(pytree_utils.slice_along_axis, axis=time_axis, idx=-1)
    self.transform_fn = transform_module(
        coords, dt, physics_specs, aux_features, input_coords
    )
    self.input_coords = input_coords
    self.modal_interpolate_fn = coordinate_systems.get_spectral_interpolate_fn(
        input_coords, 
        coords, 
        expect_same_vertical=False)

    
  def weatherbench_to_primitive(self, wb_state_nodal, surface_pressure):
    """Converts wb_state on HYBRID coordinates to primitive on SIGMA."""
    # Note: the returned values have mixed nodal/modal representations.
    regrid_fn = functools.partial(vertical_interpolation.interp_hybrid_to_sigma,
                                  hybrid_coords=vertical_interpolation.HybridCoordinates.ECMWF137(),
                                  sigma_coords=self.coords.vertical,
                                  surface_pressure=surface_pressure)  
    wb_state_on_sigma = regrid_fn(wb_state_nodal)
    u, v = self.coords.physics_to_dycore_sharding(
        (wb_state_on_sigma.u, wb_state_on_sigma.v)
    )
    vorticity, divergence = self.coords.dycore_to_physics_sharding(
        self.curl_and_div_fn(u, v)
    )
    pe_state_on_sigma = primitive_equations.StateWithTime(
        divergence=divergence,
        vorticity=vorticity,
        temperature_variation=(wb_state_on_sigma.t - self.ref_temps),
        log_surface_pressure=jnp.log(jnp.expand_dims(surface_pressure, axis=0)),
        sim_time=wb_state_on_sigma.sim_time,
        tracers=wb_state_on_sigma.tracers)
    return pe_state_on_sigma

  def __call__(self,
      inputs,
      forcing,
  ):
      del(forcing)
      #print('input', self.input_coords)
      #print('coord', self.coords)
      
      input_sliced = self.slice_fn(inputs)
      surface_pressure = input_sliced['surface_pressure'][0]
      
      wb_state = weatherbench_utils.State(**{k:input_sliced[k] for k in input_sliced if k!='surface_pressure'})
      #print('Sliced', input_sliced['surface_pressure'].shape)
      #print(surface_pressure.shape)
      #print(wb_state.t.shape)
      wb_state = coordinate_systems.maybe_to_nodal(wb_state, self.input_coords)
      pe_state = self.weatherbench_to_primitive(wb_state, surface_pressure)
      #print("after weatherbench_to_primitive")
      #print(pe_state)
      #print(pe_state.temperature_variation.shape, pe_state.log_surface_pressure.shape)
      pe_state = coordinate_systems.maybe_to_modal(pe_state, self.input_coords)
      #print("after maybe_to_modal")
      #print(pe_state.temperature_variation.shape, pe_state.log_surface_pressure.shape)
      pe_state = self.modal_interpolate_fn(pe_state)
      #print("after modal_interpolate_fn")
      #print(pe_state.temperature_variation.shape, pe_state.log_surface_pressure.shape)
      pe_state = self.transform_fn(pe_state)
      #print("after transform fn")
      #print(pe_state.temperature_variation.shape, pe_state.log_surface_pressure.shape)
      #print(pe_state.divergence.shape, pe_state.vorticity.shape)
      return encoders.ModelState(state=pe_state)

@gin.register
class DimensionalDataStateToPrimitiveEncoderWithMemory(hk.Module):
  """Encoder that converts `DataState` on the models grid to `StateWithTime`."""

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      dt: float,
      physics_specs: Any,
      aux_features: Dict[str, Any],
      input_coords: coordinate_systems.CoordinateSystem,
      inputs_to_units_mapping: Dict[str, str],
      time_axis: int = 0,
      orography_module: orographies.OrographyModule = orographies.ClippedOrography,
      transform_module: transforms.TransformModule = encoders.EncoderIdentityTransform,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.encoder_fn = DimensionalDataStateToPrimitiveEncoder(
        coords=coords,
        dt=dt,
        physics_specs=physics_specs,
        aux_features=aux_features,
        input_coords=coords,
        inputs_to_units_mapping=inputs_to_units_mapping,
        time_axis=time_axis,
        orography_module=orography_module,
        transform_module=transform_module,
        name=name,
    )

  def __call__(
      self,
      inputs: encoders.DataState,
      forcing: encoders.Forcing,
  ) -> primitive_equations.StateWithTime:

    memory = self.encoder_fn(inputs, forcing=forcing)
    model_state = self.encoder_fn(inputs, forcing=forcing)
    return encoders.ModelState(
        state=model_state.state,
        memory=memory.state,
    )
