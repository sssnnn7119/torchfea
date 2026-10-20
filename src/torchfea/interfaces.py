
import inspect

import numpy as np
import torch


class Serializable():

    _serialized_attributes: list[str] = []
    """List of attribute names to be serialized."""

    _serialized_attributes_exclude: list[str] = []
    """List of attribute names to be excluded from serialization."""

    _subclasses: dict[str, 'Serializable'] = {}
    """Registry of subclasses for factory method."""

    _subclass_source_code: dict[str, str] = {}
    """Registry of subclass source code for debugging and reproducibility."""

    def __init_subclass__(cls):
        """Register subclasses in the class registry for factory method."""
        cls._subclasses[cls._get_obj_name()] = cls
        cls._subclass_source_code[cls._get_obj_name()] = cls._get_source_code()
        excluded = []
        for klass in reversed(inspect.getmro(cls)):
            excluded.extend(klass.__dict__.get('_serialized_attributes_exclude', ()))
        cls._serialized_attributes_exclude = list(dict.fromkeys(excluded))

    @classmethod
    def _get_source_code(cls) -> str:
        """Get source code of this class and all its ancestor classes.

        Walks the MRO (Method Resolution Order) and concatenates the source
        code of each class from the root base down to this class.
        Uses ``inspect.getsource``, so the source file must be available.

        Returns:
            str: Concatenated source code of all classes in the MRO.
        """
        source_parts: list[str] = []
        source_parts.append(inspect.getsource(cls))
        return "\n".join(source_parts)

    def __init__(self) -> None:
        super().__init__()

    @property
    def serialized_attributes(self):
        """Get the list of attributes to be serialized."""
        serialized_attrs = []
        if not self._serialized_attributes:
            serialized_attrs = [attr for attr in self.__dict__.keys() if not attr.startswith('__')]
        else:
            serialized_attrs = self._serialized_attributes

        # Exclude any attributes that are in the exclude list
        serialized_attrs = [attr for attr in serialized_attrs if attr not in self._serialized_attributes_exclude]
        return serialized_attrs
    
    @classmethod
    def _get_obj_name(cls):
        """Get the name of the object's class, including the mro."""

        mro = inspect.getmro(cls)

        name: list[str] = []
        for klass in mro:
            if klass is object:
                continue
            name.append(klass.__name__)
        name = ".".join(reversed(name))
        return name

    @staticmethod
    def _serialize_obj(obj):
        """Helper function to serialize an object."""
        if isinstance(obj, torch.Tensor):
            obj_np = obj.detach().cpu().numpy()
            if obj_np.dtype == np.float64 or obj_np.dtype == np.float32:
                obj_np = obj_np.astype(np.float32)
            elif obj_np.dtype == np.int64:
                obj_np = obj_np.astype(np.int32)
            return (obj_np, type(obj).__name__)
        elif issubclass(type(obj), Serializable):
            return obj._serialize()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return (obj, type(obj).__name__)
        elif isinstance(obj, (list, tuple)):
            return ([Serializable._serialize_obj(item) for item in obj], type(obj).__name__)
        elif isinstance(obj, dict):
            return ({key: Serializable._serialize_obj(value) for key, value in obj.items()}, type(obj).__name__)
        elif isinstance(obj, np.ndarray):
            return (obj, type(obj).__name__)
        else:
            raise TypeError(
                f"Cannot serialize object of type {type(obj).__module__}."
                f"{type(obj).__qualname__}.")

    def _serialize(self) -> dict:
        """
        Serialize the object to a dictionary.

        Returns:
            dict: A dictionary containing the serialized attributes.
        """


        serialized_data = {}

        all_attributes = [attr for attr in self.__dict__.keys() if not attr.startswith('__')]
        selected_attributes = self.serialized_attributes
        
        selected_attributes = list(set(selected_attributes))

        for attr in all_attributes:
            if attr in selected_attributes:
                value = getattr(self, attr)
                sub_serialized = self._serialize_obj(value)
                if sub_serialized:
                    serialized_data[attr] = sub_serialized
            else:
                serialized_data[attr] = (None, 'NoneType')


        return (serialized_data, self._get_obj_name())
    
    @staticmethod
    def _deserialize_obj(data: tuple):
        """Helper function to deserialize an object."""
        if not data:
            return None
        value, type_name = data
        if type_name in Serializable._subclasses:
            return Serializable._subclasses[type_name]._deserialize((value, type_name))
        elif type_name == 'Tensor':
            data_now = torch.from_numpy(value).to(torch.get_default_device())
            if data_now.dtype == torch.float64 or data_now.dtype == torch.float32:
                data_now = data_now.to(torch.get_default_dtype())
            return data_now
        elif type_name in ['int', 'float', 'str', 'bool', 'NoneType']:
            return value
        elif type_name in ['list', 'tuple']:
            return [Serializable._deserialize_obj(item) for item in value]
        elif type_name == 'dict':
            return {key: Serializable._deserialize_obj(val) for key, val in value.items()}
        elif type_name == 'ndarray':
            return value
        else:
            raise ValueError(f"Unknown type name '{type_name}' during deserialization.")
    
    @classmethod
    def _deserialize(cls, data: tuple[dict, str]):
        """
        Deserialize the object from a dictionary.

        Args:
            data (dict): A dictionary containing the serialized attributes.
        """
        serialized_data, class_name = data
        if class_name not in cls._subclasses:
            raise ValueError(f"Unknown class name '{class_name}' during deserialization.")
        
        obj = cls._subclasses[class_name].__new__(cls._subclasses[class_name])
        for attr, value in serialized_data.items():
            # This field was persisted by older versions, but it is now a
            # runtime property derived by each object from its current state.
            if attr == "_RGC_requirements":
                continue
            deserialized_value = cls._deserialize_obj(value)
            setattr(obj, attr, deserialized_value)
        return obj
    
Serializable._subclass_source_code['Serializable'] = inspect.getsource(Serializable)
