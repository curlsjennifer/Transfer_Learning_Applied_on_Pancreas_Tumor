import hashlib


def get_config_sha1(config, digit=5):
    """Get the sha1 of configuration for Experiment ID

    config will be converted to str and sha.

    Args:
        config (dict): The dictionary contains configuration information.
        digit (int, optional): The number of starting digit. Defaults to 5.

    Returns:
        str: First "digit" of config's sha1

    """
    s = hashlib.sha1()
    s.update(str(config).encode('utf-8'))
    return s.hexdigest()[:digit]


def count_parameters(model):
    """Get the number of trainable params

    Parameters is trainable iff it requires gradient.

    Args:
        model (pytorch model): The pytorch model.

    Returns:
        int: number of trainable parameters

    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
