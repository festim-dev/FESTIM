from fenics import *
import csv
import sys
import os
import sympy as sp
import json
import numpy as np


def write_to_csv(dict, desorption):
    if "file" in dict.keys():
        file_export = ''
        if "folder" in dict.keys():
            file_export += dict["folder"] + '/'
            os.makedirs(os.path.dirname(file_export), exist_ok=True)
        if dict["file"].endswith(".csv"):
            file_export += dict["file"]
        else:
            file_export += dict["file"] + ".csv"
        busy = True
        while busy is True:
            try:
                with open(file_export, "w+") as f:
                    busy = False
                    writer = csv.writer(f, lineterminator='\n')
                    for val in desorption:
                        writer.writerows([val])
            except OSError as err:
                print("OS error: {0}".format(err))
                print("The file " + file_export + ".txt might currently be busy."
                      "Please close the application then press any key.")
                input()

    return True


def export_txt(filename, function, W):
    '''
    Exports a 1D function into a txt file.
    Arguments:
    - filemame : str
    - function : FEniCS Function
    - W : FunctionSpace on which the solution will be projected.
    Returns:
    - True on sucess,
    - False on failure
    '''
    export = Function(W)
    export = project(function)
    busy = True
    x = interpolate(Expression('x[0]', degree=1), W)
    while busy is True:
        try:
            np.savetxt(filename + '.txt', np.transpose(
                        [x.vector()[:], export.vector()[:]]))
            return True
        except OSError as err:
            print("OS error: {0}".format(err))
            print("The file " + filename + ".txt might currently be busy."
                  "Please close the application then press any key.")
            input()
    return False


def export_profiles(res, exports, t, dt, W):
    '''
    Exports 1D profiles in txt files.
    Arguments:
    - res: list, contains FEniCS Functions
    - exports: dict, dict, contains parameters
    - t: float, time
    - dt: FEniCS Constant(), stepsize
    Returns:
    - dt: FEniCS Constant(), stepsize
    '''
    functions = exports['txt']['functions']
    labels = exports['txt']['labels']
    if len(functions) != len(labels):
        raise NameError("Number of functions to be exported "
                        "doesn't match number of labels in txt exports")
    if len(functions) > len(res):
        raise NameError("Too many functions to export "
                        "in txt exports")
    solution_dict = {
        'solute': res[0],
        'retention': res[len(res)-2],
        'T': res[len(res)-1],
    }
    times = sorted(exports['txt']['times'])
    end = True
    for time in times:
        if t == time:
            if times.index(time) != len(times)-1:
                next_time = times[times.index(time)+1]
                end = False
            else:
                end = True
            for i in range(len(functions)):
                if functions[i].isdigit() is True:
                    solution = res[int(functions[i])]
                elif functions[i] in solution_dict:
                    solution = solution_dict[functions[i]]
                else:
                    raise ValueError(
                        "function " + functions[i] + " is unknown")
                label = labels[i]
                export_txt(
                    exports["txt"]["folder"] + '/' + label + '_' +
                    str(t) + 's',
                    solution, W)
            break
        if t < time:
            next_time = time
            end = False
            break
    if end is False:
        if t + float(dt) > next_time:
            dt.assign(time - t)
    return dt


def define_xdmf_files(exports):
    '''
    Returns a list of XDMFFile
    Arguments:
    - exports: dict, contains parameters
    '''
    if len(exports['xdmf']['functions']) != len(exports['xdmf']['labels']):
        raise NameError("Number of functions to be exported "
                        "doesn't match number of labels in xdmf exports")
    if exports["xdmf"]["folder"] == "":
        raise ValueError("folder value cannot be an empty string")
    if type(exports["xdmf"]["folder"]) is not str:
        raise TypeError("folder value must be of type str")
    files = list()
    for i in range(0, len(exports["xdmf"]["functions"])):
        u_file = XDMFFile(exports["xdmf"]["folder"]+'/' +
                          exports["xdmf"]["labels"][i] + '.xdmf')
        u_file.parameters["flush_output"] = True
        u_file.parameters["rewrite_function_mesh"] = False
        files.append(u_file)
    return files


def export_xdmf(res, exports, files, t, append):
    '''
    Exports the solutions fields in xdmf files.
    Arguments:
    - res: list, contains FEniCS Functions
    - exports: dict, contains parameters
    - files: list, contains XDMFFile
    - t: float
    - append: bool, erase the previous file or not
    '''
    if len(exports['xdmf']['functions']) > len(res):
        raise NameError("Too many functions to export "
                        "in xdmf exports")
    solution_dict = {
        'solute': res[0],
        'retention': res[len(res)-2],
        'T': res[len(res)-1],
    }
    for i in range(0, len(exports["xdmf"]["functions"])):
        label = exports["xdmf"]["labels"][i]
        fun = exports["xdmf"]["functions"][i]
        if type(fun) is int:
            if fun <= len(res):
                solution = res[fun]
            else:
                raise ValueError(
                    "The value " + str(fun) +
                    " is unknown.")
        elif type(fun) is str:
            if fun.isdigit():
                fun = int(fun)
                if fun <= len(res):
                    solution = res[fun]
                else:
                    raise ValueError(
                        "The value " + str(fun) +
                        " is unknown.")
            elif fun in solution_dict.keys():
                solution = solution_dict[fun]
            else:
                raise ValueError(
                    "The value " + fun +
                    " is unknown.")
        else:
            raise TypeError('Unexpected' + str(type(fun)) + 'type')

        solution.rename(label, "label")
        # files[i].write_checkpoint(
        #     solution, label, t, XDMFFile.Encoding.HDF5, append=append)
        files[i].write(solution, t)
    return


def treat_value(d):
    '''
    Recursively converts as string the sympy objects in d
    Arguments: d, dict
    '''
    T = sp.symbols('T')
    if type(d) is dict:
        for key, value in d.items():
            if isinstance(value, tuple(sp.core.all_classes)):
                value = str(sp.printing.ccode(value))
                d[key] = value
            elif callable(value):  # if value is fun
                d[key] = str(sp.printing.ccode(value(T)))
            elif type(value) is dict or type(value) is list:
                d[key] = treat_value(value)
    elif type(d) is list:
        for e in d:
            e = treat_value(e)
    return d


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
