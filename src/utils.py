import hashlib
from keras import backend as K


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


def f1_keras_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

