import yaml
import pickle


def read_yaml(file_path):
    """
    Read a YAML file and return the contents as a dictionary.
    Parameters
    ----------
    file_path : str
        The path of the YAML file to read.
    Returns
    -------
    dict
        The contents of the YAML file as a dictionary.
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
        return data


def write_yaml(data, file_path):
    """
    Write a dictionary to a YAML file.
    Parameters
    ----------
    data : dict
        The dictionary to write to a YAML file.
    file_path : str
        The path of the file where the YAML data shall be stored.
    Returns
    -------
    None
    """
    with open(file_path, 'w') as f:
        yaml.safe_dump(data, f)


def update_yaml(file_path, new_items):
    """
    Update a YAML file with the contents of a dictionary.
    Parameters
    ----------
    file_path : str
        The path of the YAML file to update.
    new_items : dict
        The dictionary of new key-value pairs to add to the existing YAML file.
    Returns
    -------
    None
    """
    data = read_yaml(file_path)
    data.update(new_items)
    write_yaml(data, file_path)
   
    
def pickle_to_dict(file_path):
    """
    Loads a dictionary from a pickle file at a specified file path.
    Parameters
    ----------
    file_path : str
        The path of the pickle file to load.
    Returns
    -------
    dict
        The dictionary loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        dic = pickle.load(f)
    return dic
    
    
def drop_keys(dict, keys_to_drop):
    """Removes specified keys from a dictionary.
    Parameters
    ----------
    dict : dict
        The dictionary to remove keys from.
    keys_to_drop : list
        A list of keys to remove from the dictionary.
    Returns
    -------
    dict
        A modified dictionary with the specified keys removed.
    """
    # Create a set of keys to be dropped
    keys_to_drop_set = set(keys_to_drop)
    # Create a new dictionary with all elements from dict except for the ones in keys_to_drop
    new_dict = {key: dict[key] for key in dict.keys() if key not in keys_to_drop_set}
    return new_dict
    