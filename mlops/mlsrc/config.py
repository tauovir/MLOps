import json
def get_configurations ( config_file : str) -> dict:
    """
    Description: This function read json file and return the ison contents
    param:
    config_file : json file path : ie :/dbfs/mnt/dir/ config. j son
    """
    try:
        with open ( config_file, mode='r') as f:
            conf = json. load (f)
    except FileNotFoundError:
        raise ValueError (f"Invailid config file name:{config_file}")
    return conf 

    