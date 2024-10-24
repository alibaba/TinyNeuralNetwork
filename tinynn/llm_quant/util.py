import ctypes
import os
import platform
import sys
import importlib

import torch.nn as nn


def _init_patch_easyquant():
    pkg_root = os.path.dirname(
        os.path.realpath(importlib.machinery.PathFinder().find_module("easyquant").get_filename())
    )
    libs_dir = os.path.abspath(pkg_root)
    is_conda_cpython = platform.python_implementation() == 'CPython' and (
        hasattr(ctypes.pythonapi, 'Anaconda_GetVersion') or 'packaged by conda-forge' in sys.version
    )
    if sys.version_info[:2] >= (3, 8) and not is_conda_cpython or sys.version_info[:2] >= (3, 10):
        if os.path.isdir(libs_dir):
            os.add_dll_directory(libs_dir)
    else:
        load_order_filepath = os.path.join(libs_dir, '.load-order-easyquant-0.0.1')
        if os.path.isfile(load_order_filepath):
            with open(load_order_filepath, 'r', encoding='utf-8') as file:
                load_order = file.read().split()
            for lib in load_order:
                lib_path = os.path.join(os.path.join(libs_dir, lib))
                if os.path.isfile(lib_path) and not ctypes.windll.kernel32.LoadLibraryExW(
                    ctypes.c_wchar_p(lib_path), None, 0x00000008
                ):
                    raise OSError('Error loading {}; {}'.format(lib, ctypes.FormatError()))


def get_submodule_with_parent_from_name(model, module_name):
    """Gets the submodule with its parent and sub_name using the name given"""
    module_name_parts = module_name.split('.')
    cur_obj = model
    last_obj = None

    for ns in module_name_parts:
        last_obj = cur_obj
        if type(cur_obj) is nn.ModuleList:
            cur_obj = cur_obj[int(ns)]
        elif type(cur_obj) is nn.ModuleDict:
            cur_obj = cur_obj[ns]
        else:
            cur_obj = getattr(cur_obj, ns)

    return cur_obj, last_obj, module_name_parts[-1]
