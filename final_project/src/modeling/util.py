import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def get_class_weight(y):

    if y is None:
        raise RuntimeError("Impossible to calculate class weight because y is empty")

    #print("Calculating class weights")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    #print("Labels:", np.unique(y))
    #print("Class weights:", class_weights)
    d_class_weights = dict(enumerate(class_weights))

    return d_class_weights
