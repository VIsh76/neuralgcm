from dinosaur import coordinate_systems
from dinosaur import xarray_utils
import gcsfs
import gin
import jax
import neuralgcm
import numpy as np
import pickle
from typing import Any
import xarray

# Keep these imports to register the encoders/decoders with gin
from . import sigma_level_encoders
from . import sigma_level_decoders


_FIELD_TO_UNITS_MAPPING = \
    {'sim_time': 'dimensionless',
     'u_component_of_wind': 'meter / second',
     'v_component_of_wind': 'meter / second',
     'temperature': 'kelvin',
     'surface_pressure': 'millibar',
     'specific_humidity': 'dimensionless',
     'specific_cloud_ice_water_content': 'dimensionless',
     'specific_cloud_liquid_water_content': 'dimensionless'}


def load_checkpoint(model_path, model_name):
    if model_path.startswith("gs:"):
        gcs = gcsfs.GCSFileSystem(token='anon')
        with gcs.open(f'{model_path}/{model_name}.pkl', 'rb') as f:
            ckpt = pickle.load(f)
    else:
        with open(f'{model_path}/{model_name}.pkl', 'rb') as f:
            ckpt = pickle.load(f)
    return ckpt


@jax.tree_util.register_pytree_node_class
class SigmaLevelModel:
    def __init__(
            self,
            structure: neuralgcm.model_builder.WhirlModel,
            params: Any,
            gin_config: str
    ):
        self._plm = neuralgcm.PressureLevelModel(structure, params, gin_config)
        self.gin_config = gin_config
        self._plm._input_variables.remove("geopotential")
        self._plm._input_variables.append("surface_pressure")

    
    def tree_flatten(self):
        leaves, params_def = jax.tree_util.tree_flatten(self._plm.params)
        return (leaves, (params_def, self._plm._structure, self._plm.gin_config))


    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        params_def, structure, gin_config = aux_data
        params = jax.tree_util.tree_unflatten(params_def, leaves)
        return cls(structure, params, gin_config)
    

    @property
    def model_coords(self) -> coordinate_systems.CoordinateSystem:
        return self._plm.model_coords
    

    @property
    def data_coords(self) -> coordinate_systems.CoordinateSystem:
        return self._plm.model_coords
    

    @property
    def input_variables(self) -> coordinate_systems.CoordinateSystem:
        return self._plm.input_variables
    

    @property
    def forcing_variables(self) -> coordinate_systems.CoordinateSystem:
        return self._plm.forcing_variables

   
    def inputs_from_xarray(self, dataset: xarray.Dataset) -> dict[str, np.ndarray]:
        input_vars = self._plm._input_variables
        neuralgcm.api._check_variables(
            dataset,
            desired_level_variables=[var for var in input_vars if var != 'surface_pressure'],
            desired_surface_variables=['surface_pressure']
        )
        return self._plm._data_from_xarray(dataset, input_vars)


    def forcings_from_xarray(self, dataset: xarray.Dataset) -> dict[str, np.ndarray]:
        return self._plm.forcings_from_xarray(dataset)


    def data_to_xarray(
            self,
            data: dict[str, neuralgcm.api.ArrayLike],
            times: np.ndarray | None, decoded: bool = True) -> xarray.Dataset:
        return self._plm.data_to_xarray(data, times, decoded=False)
    

    @jax.jit
    @neuralgcm.api._static_gin_config
    def encode(self, inputs, forcings, rng_key):
        sim_time = inputs['sim_time']
        inputs = self._plm._to_abbreviated_names_and_tracers(inputs)
        inputs = neuralgcm.api._prepend_dummy_time_axis(inputs)
        forcings = self._plm._squeeze_level_from_forcings(forcings)
        forcings = neuralgcm.api._prepend_dummy_time_axis(forcings)
        f = self._plm._structure.forcing_fn(self._plm.params, None, forcings, sim_time)
        return self._plm._structure.encode_fn(self._plm.params, rng_key, inputs, f)
    

    @jax.jit
    @neuralgcm.api._static_gin_config
    def decode(self, state, forcings):
        sim_time = neuralgcm.api._sim_time_from_state(state)
        forcings = self._plm._squeeze_level_from_forcings(forcings)
        forcings = neuralgcm.api._prepend_dummy_time_axis(forcings)
        f = self._plm._structure.forcing_fn(self._plm.params, None, forcings, sim_time)
        outputs = self._plm._structure.decode_fn(self._plm.params, None, state, f)
        return outputs
        

    def advance(self, model_state, forcings):
        return self._plm.advance(model_state, forcings)


    def unroll(
            self,
            model_state,
            forcings,
            *,
            steps,
            timedelta,
            start_with_input
            ):
        post_process_fn = self.decode
        return self._plm.unroll(
            model_state,
            forcings,
            steps=steps,
            timedelta=timedelta,
            start_with_input=start_with_input,
            post_process_fn=post_process_fn)


    @classmethod
    def from_checkpoint(cls, checkpoint: Any):
        """
        Creates a SigmaLevelModel from a checkpoint.
        """

        params = checkpoint['params']
        gin_config = checkpoint['model_config_str']
        aux_ds_dict = checkpoint['aux_ds_dict']

        with neuralgcm.gin_utils.specific_config(gin_config):
            
            physics_specs = neuralgcm.physics_specifications.get_physics_specs()
            aux_ds = xarray.Dataset.from_dict(aux_ds_dict)
            data_coords = neuralgcm.model_builder.coordinate_system_from_dataset(aux_ds)
            model_specs = neuralgcm.model_builder.get_model_specs(
                data_coords, physics_specs, {xarray_utils.XARRAY_DS_KEY: aux_ds}
            )
            whirl_model = neuralgcm.model_builder.WhirlModel(
                coords=model_specs.coords,
                dt=model_specs.dt,
                physics_specs=model_specs.physics_specs,
                aux_features=model_specs.aux_features,
                input_coords=data_coords,
                output_coords=data_coords,
            )
            
            return cls(whirl_model, params, gin_config)
    
    
    @classmethod
    def from_plm_checkpoint(cls, checkpoint: Any):
        """
        Creates a SigmaLevelModel from a PressureLevelModel checkpoint.
        """
        modified_checkpoint = checkpoint.copy()

        # Modify gin config
        with neuralgcm.gin_utils.specific_config(modified_checkpoint['model_config_str']):
            model_cls_ref = gin.query_parameter('WhirlModel.model_cls')
            step_cls_ref = gin.query_parameter(
                model_cls_ref.scoped_selector+'.advance_module')
            
            if step_cls_ref.scoped_selector == 'StochasticPhysicsParameterizationStep':
                gin.bind_parameter(
                    'DimensionalDataStateToPrimitiveEncoderWithMemory.inputs_to_units_mapping', _FIELD_TO_UNITS_MAPPING)
                
                gin.bind_parameter(
                    'DimensionalPrimitiveToDataStateDecoder.inputs_to_units_mapping', _FIELD_TO_UNITS_MAPPING)
                
                gin.parse_config(
                    model_cls_ref.scoped_selector+'.encoder_module = @DimensionalDataStateToPrimitiveEncoderWithMemory')
                
                gin.parse_config(
                    model_cls_ref.scoped_selector+'.decoder_module = @DimensionalPrimitiveToDataStateDecoder')
                
                modified_checkpoint['model_config_str'] = gin.config_str()
            else:
                raise ValueError(f"Unsupported step_cls '{step_cls_ref.scoped_selector}'.")

        return cls.from_checkpoint(modified_checkpoint)
