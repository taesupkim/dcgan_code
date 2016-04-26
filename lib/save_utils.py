import os
import tempfile
import cPickle
import sys
import shutil
from collections import OrderedDict

def save_model(tensor_params_list, save_to):
    tensor_params_dict = get_tensor_params(tensor_params_list)
    params_value_dict  = get_tensor_params_val(tensor_params_dict)
    secure_pickle_dump(params_value_dict, save_to)


def get_tensor_params(tensor_params_list):
    tensor_params_dict = OrderedDict()
    for param_tensor in tensor_params_list:
        tensor_params_dict[param_tensor.name] = param_tensor
    return tensor_params_dict

def get_tensor_params_val(tensor_params_dict):
    params_dict = OrderedDict()
    for param_name, param_tensor in tensor_params_dict.iteritems():
        params_dict[param_name] = param_tensor.get_value()
    return params_dict

def secure_pickle_dump(object_, path):
    """
    This code is brought from Blocks.
    Robust serialization - does not corrupt your files when failed.
    Parameters
    ----------
    object_ : object
        The object to be saved to the disk.
    path : str
        The destination path.
    """
    try:
        d = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(delete=False, dir=d) as temp:
            try:
                cPickle.dump(object_, temp, protocol=-1)
            except RuntimeError as e:
                if str(e).find('recursion') != -1:
                    Warning('cle.utils.secure_pickle_dump encountered '
                        'the following error: ' + str(e) +
                        '\nAttempting to resolve this error by calling ' +
                        'sys.setrecusionlimit and retrying')
                    old_limit = sys.getrecursionlimit()
                try:
                    sys.setrecursionlimit(50000)
                    cPickle.dump(object_, temp, protocol=-1)
                finally:
                    sys.setrecursionlimit(old_limit)
        shutil.move(temp.name, path)
    except:
        if "temp" in locals():
            os.remove(temp.name)
        raise
