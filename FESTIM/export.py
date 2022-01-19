import os
import sympy as sp
import json


class Export:
    def __init__(self, field=None) -> None:
        self.field = field
        self.function = None

class Exports:
    def __init__(self, exports=[]) -> None:
        self.exports = exports


def treat_value(d):
    '''
    Recursively converts as string the sympy objects in d
    Arguments: d, dict
    Returns: d, dict
    '''

    T = sp.symbols('T')
    if type(d) is dict:
        d2 = {}
        for key, value in d.items():
            if isinstance(value, tuple(sp.core.all_classes)):
                value = str(sp.printing.ccode(value))
                d2[key] = value
            elif callable(value):  # if value is fun
                d2[key] = str(sp.printing.ccode(value(T)))
            elif type(value) is dict or type(value) is list:
                d2[key] = treat_value(value)
            else:
                d2[key] = value
    elif type(d) is list:
        d2 = []
        for e in d:
            e2 = treat_value(e)
            d2.append(e2)
    else:
        d2 = d

    return d2


def export_parameters(parameters):
    '''
    Dumps parameters dict in a json file.
    '''
    json_file = parameters["exports"]["parameters"]
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    if json_file.endswith(".json") is False:
        json_file += ".json"
    param = treat_value(parameters)
    with open(json_file, 'w') as fp:
        json.dump(param, fp, indent=4, sort_keys=True)
    return True
