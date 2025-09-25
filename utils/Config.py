import yaml


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # If the value is a dictionary, recursively convert it into another Config object
                value = Config(value)
            setattr(self, key, value)

    def __repr__(self):
        return f'{self.__dict__}'


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


