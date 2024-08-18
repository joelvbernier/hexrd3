import numpy as np
import yaml


class NumpyToNativeDumper(yaml.SafeDumper):
    """Change Numpy types to native types during YAML encoding

    This inherits from yaml.SafeDumper so that anything that is not
    converted to a basic type will raise an error.

    For instance, np.float128 will raise an error, since it cannot be
    converted to a basic type.
    """
    def represent_data(self, data):
        # Empty shape arrays should be treated as numbers, not arrays.
        is_empty_shape_array = (
            isinstance(data, np.ndarray) and data.shape == ()
        )
        if isinstance(data, np.ndarray) and not is_empty_shape_array:
            return self.represent_list(data.tolist())
        elif isinstance(data, (np.generic, np.number)) or is_empty_shape_array:
            item = data.item()
            if isinstance(item, (np.generic, np.number)):
                # This means it was not converted successfully.
                # It is probably np.float128.
                msg = (
                    f'Failed to convert {item} with type {type(item)} to '
                    'a native type'
                )
                raise yaml.representer.RepresenterError(msg)

            return self.represent_data(item)

        return super().represent_data(data)
