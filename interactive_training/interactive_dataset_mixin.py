from datasets import IterableDataset
from torch.utils.data import Dataset
from transformers.utils import logging
from typing import Generic, TypeVar, List, Dict
from interactive_training.constants import RESERVED_NAMES, BASIC_TYPES, TYPE_MAPPING

DatasetType = TypeVar("TrainerType", Dataset, IterableDataset)


logger = logging.get_logger(__name__)


class InteractiveDatasetMixin(Generic[DatasetType]):
    def __init__(
        self,
        interactive_parameter_names: List[str],
        **kwargs,
    ):
        """
        Initialize the interactive dataset mixin.
        :param interactive_paramaters: List of parameters that can be updated interactively.
        :param kwargs: Additional keyword arguments for the dataset.
        """

        self._data_kwargs = kwargs

        super().__init__(**kwargs)

        self._interactive_parameter_names = interactive_parameter_names

        self.initializataion_parameters = (
            self.get_updateable_initialization_parameters()
        )

        self._cmd_queue = None
        self._control_queue = None

        self.interactive_parameters = self.get_updateable_interactive_parameters()

    def _check_nested_type(self, value: dict | tuple | list | set) -> bool:
        """
        Check if the value is a nested type containing only basic types.
        Supported nested types are: list, dict, tuple, and set.
        Each element in the nested structure must be of a basic type: int, float, str, bool.
        If the value is a dict, both keys and values must be of basic types.
        """

        if isinstance(value, dict):
            return all(isinstance(v, BASIC_TYPES) for v in value.values()) and all(
                isinstance(k, BASIC_TYPES) for k in value.keys()
            )
        elif isinstance(value, (list, tuple, set)):
            return all(isinstance(v, BASIC_TYPES) for v in value)
        return False

    def _check_type(self, value: any) -> bool:
        """
        Check if the value is of a supported type.
        Supported types are: int, float, str, bool, list, dict, and tuple.
        """
        return isinstance(value, BASIC_TYPES) or self._check_nested_type(value)

    def update_runtime_parameters(self, new_parameters: Dict[str, any]):
        success = True
        for param_name, value in new_parameters.items():
            if param_name in self._interactive_parameter_names:
                if self._check_type(value):
                    setattr(self, param_name, value)
                    logger.info(f"Parameter {param_name} updated to {value}.")
                    self.interactive_parameters[param_name] = value
                else:
                    logger.warning(
                        f"Parameter {param_name} has an unsupported type: {type(value)}. "
                        "Supported types are: int, float, str, bool, list, dict, and tuple."
                    )
                    success = False
            else:
                print(f"Parameter {param_name} is not an interactive parameter.")
                success = False
        return success

    def update_intialization_parameters(self, new_parameters: Dict[str, any]):
        """
        Update the initialization parameters of the dataset.
        This method updates the _data_kwargs with new parameters.
        """

        success = True

        for param_name, value in new_parameters.items():
            if param_name in self._data_kwargs:
                if self._check_type(value):
                    self._data_kwargs[param_name] = value
                    self.initializataion_parameters[param_name] = value
                    logger.info(
                        f"Initialization parameter {param_name} updated to {value}."
                    )
                else:
                    logger.warning(
                        f"Initialization parameter {param_name} has an unsupported type: {type(value)}. "
                        "Supported types are: int, float, str, bool, list, dict, and tuple."
                    )
                    success = False
            else:
                logger.warning(f"Initialization parameter {param_name} does not exist.")
                success = False

        return success

    def get_updateable_interactive_parameters(self) -> Dict[str, str]:
        """
        Collect parameters from the dataset.
        return the name and types of the parameters.
        We only support basic types: int, float, str, bool, list, dict, and tuple.
        If the parameter is a nested type (e.g., a list of dicts), it will be checked
        to ensure all elements are of basic types.
        """
        params = {}
        for param in self._interactive_parameter_names:
            if param not in RESERVED_NAMES and hasattr(self, param):
                if self._check_type(getattr(self, param)):
                    logger.info(
                        f"Interactive  Parameter {param} has type {type(getattr(self, param)).__name__} and value {getattr(self, param)}."
                    )
                    params[param] = getattr(self, param)
                else:
                    logger.warning(
                        f"Interactive  Parameter {param} has an unsupported type: {type(getattr(self, param))}. "
                        "Supported types are: int, float, str, bool, list, dict, and tuple."
                    )
                    continue
            else:
                logger.warning(
                    f"Interactive Parameter {param} does not exist in the dataset."
                )
                continue
        return params

    def get_updateable_initialization_parameters(self) -> Dict[str, str]:
        """
        Collect parameters that can be updated interactively from the initialization parameters.
        This method returns a dictionary of parameter names and their types.
        """
        params = {}
        for param, value in self._data_kwargs.items():
            if param not in RESERVED_NAMES and self._check_type(value):
                logger.info(
                    f"Initialization Parameter {param} has type {type(value).__name__} and value {value}."
                )
                params[param] = value
            else:
                logger.warning(
                    f"Initialization Parameter {param} does not exist in the dataset."
                )
        return params

    def reload_dataset(self):
        updated_kwargs = {}
        for param_name, value in self._data_kwargs.items():
            if param_name in self.initializataion_parameters:
                updated_kwargs[param_name] = self.initializataion_parameters[param_name]
            else:
                updated_kwargs[param_name] = value

        if hasattr(super(), "reload_dataset"):
            super().reload_dataset(**updated_kwargs)
        else:
            super().__init__(**updated_kwargs)
            for param_name, value in self.interactive_parameters.items():
                if hasattr(self, param_name):
                    setattr(self, param_name, value)
                else:
                    logger.warning(
                        f"Interactive Parameter {param_name} does not exist in the dataset."
                    )
