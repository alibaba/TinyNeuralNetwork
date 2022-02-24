# The following code is based on forbiddenfruit.
# URL: https://github.com/clarete/forbiddenfruit
#
# Copyright (c) 2013-2020  Lincoln de Sousa <lincoln@clarete.li>
#
# This program is licensed under MIT.
#
# MIT
# ---
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import ctypes
from functools import wraps

import builtins as __builtin__

Py_ssize_t = ctypes.c_int64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_int32


class PyObject(ctypes.Structure):
    def incref(self):
        self.ob_refcnt += 1

    def decref(self):
        self.ob_refcnt -= 1


class PyFile(ctypes.Structure):
    pass


PyObject_p = ctypes.py_object
Inquiry_p = ctypes.CFUNCTYPE(ctypes.c_int, PyObject_p)
# return type is void* to allow ctypes to convert python integers to
# plain PyObject*
UnaryFunc_p = ctypes.CFUNCTYPE(ctypes.py_object, PyObject_p)
BinaryFunc_p = ctypes.CFUNCTYPE(ctypes.py_object, PyObject_p, PyObject_p)
TernaryFunc_p = ctypes.CFUNCTYPE(ctypes.py_object, PyObject_p, PyObject_p, PyObject_p)
LenFunc_p = ctypes.CFUNCTYPE(Py_ssize_t, PyObject_p)
SSizeArgFunc_p = ctypes.CFUNCTYPE(ctypes.py_object, PyObject_p, Py_ssize_t)
SSizeObjArgProc_p = ctypes.CFUNCTYPE(ctypes.c_int, PyObject_p, Py_ssize_t, PyObject_p)
ObjObjProc_p = ctypes.CFUNCTYPE(ctypes.c_int, PyObject_p, PyObject_p)
ObjObjArgProc_p = ctypes.CFUNCTYPE(ctypes.c_int, PyObject_p, PyObject_p, PyObject_p)

FILE_p = ctypes.POINTER(PyFile)


def get_not_implemented():
    namespace = {}
    name = "_Py_NotImplmented"
    not_implemented = ctypes.cast(ctypes.pythonapi._Py_NotImplementedStruct, ctypes.py_object)

    ctypes.pythonapi.PyDict_SetItem(ctypes.py_object(namespace), ctypes.py_object(name), not_implemented)
    return namespace[name]


# address of the _Py_NotImplementedStruct singleton
NotImplementedRet = get_not_implemented()


class PyNumberMethods(ctypes.Structure):
    _fields_ = [
        ('nb_add', BinaryFunc_p),
        ('nb_subtract', BinaryFunc_p),
        ('nb_multiply', BinaryFunc_p),
        ('nb_remainder', BinaryFunc_p),
        ('nb_divmod', BinaryFunc_p),
        ('nb_power', BinaryFunc_p),
        ('nb_negative', UnaryFunc_p),
        ('nb_positive', UnaryFunc_p),
        ('nb_absolute', UnaryFunc_p),
        ('nb_bool', Inquiry_p),
        ('nb_invert', UnaryFunc_p),
        ('nb_lshift', BinaryFunc_p),
        ('nb_rshift', BinaryFunc_p),
        ('nb_and', BinaryFunc_p),
        ('nb_xor', BinaryFunc_p),
        ('nb_or', BinaryFunc_p),
        ('nb_int', UnaryFunc_p),
        ('nb_reserved', ctypes.c_void_p),
        ('nb_float', UnaryFunc_p),
        ('nb_inplace_add', BinaryFunc_p),
        ('nb_inplace_subtract', BinaryFunc_p),
        ('nb_inplace_multiply', BinaryFunc_p),
        ('nb_inplace_remainder', BinaryFunc_p),
        ('nb_inplace_power', TernaryFunc_p),
        ('nb_inplace_lshift', BinaryFunc_p),
        ('nb_inplace_rshift', BinaryFunc_p),
        ('nb_inplace_and', BinaryFunc_p),
        ('nb_inplace_xor', BinaryFunc_p),
        ('nb_inplace_or', BinaryFunc_p),
        ('nb_floor_divide', BinaryFunc_p),
        ('nb_true_divide', BinaryFunc_p),
        ('nb_inplace_floor_divide', BinaryFunc_p),
        ('nb_inplace_true_divide', BinaryFunc_p),
        ('nb_index', BinaryFunc_p),
        ('nb_matrix_multiply', BinaryFunc_p),
        ('nb_inplace_matrix_multiply', BinaryFunc_p),
    ]


class PySequenceMethods(ctypes.Structure):
    _fields_ = [
        ('sq_length', LenFunc_p),
        ('sq_concat', BinaryFunc_p),
        ('sq_repeat', SSizeArgFunc_p),
        ('sq_item', SSizeArgFunc_p),
        ('was_sq_slice', ctypes.c_void_p),
        ('sq_ass_item', SSizeObjArgProc_p),
        ('was_sq_ass_slice', ctypes.c_void_p),
        ('sq_contains', ObjObjProc_p),
        ('sq_inplace_concat', BinaryFunc_p),
        ('sq_inplace_repeat', SSizeArgFunc_p),
    ]


class PyMappingMethods(ctypes.Structure):
    _fields_ = [
        ('mp_length', LenFunc_p),
        ('mp_subscript', BinaryFunc_p),
        ('mp_ass_subscript', ObjObjArgProc_p),
    ]


class PyTypeObject(ctypes.Structure):
    pass


class PyAsyncMethods(ctypes.Structure):
    pass


PyObject._fields_ = [
    ('ob_refcnt', Py_ssize_t),
    ('ob_type', ctypes.POINTER(PyTypeObject)),
]

PyTypeObject._fields_ = [
    # varhead
    ('ob_base', PyObject),
    ('ob_size', Py_ssize_t),
    # declaration
    ('tp_name', ctypes.c_char_p),
    ('tp_basicsize', Py_ssize_t),
    ('tp_itemsize', Py_ssize_t),
    ('tp_dealloc', ctypes.CFUNCTYPE(None, PyObject_p)),
    ('printfunc', ctypes.CFUNCTYPE(ctypes.c_int, PyObject_p, FILE_p, ctypes.c_int)),
    ('getattrfunc', ctypes.CFUNCTYPE(PyObject_p, PyObject_p, ctypes.c_char_p)),
    ('setattrfunc', ctypes.CFUNCTYPE(ctypes.c_int, PyObject_p, ctypes.c_char_p, PyObject_p)),
    ('tp_as_async', ctypes.CFUNCTYPE(PyAsyncMethods)),
    ('tp_repr', ctypes.CFUNCTYPE(PyObject_p, PyObject_p)),
    ('tp_as_number', ctypes.POINTER(PyNumberMethods)),
    ('tp_as_sequence', ctypes.POINTER(PySequenceMethods)),
    ('tp_as_mapping', ctypes.POINTER(PyMappingMethods)),
    ('tp_hash', ctypes.CFUNCTYPE(ctypes.c_int64, PyObject_p)),
    ('tp_call', ctypes.CFUNCTYPE(PyObject_p, PyObject_p, PyObject_p, PyObject_p)),
    ('tp_str', ctypes.CFUNCTYPE(PyObject_p, PyObject_p)),
    # ...
]

_decref = ctypes.pythonapi.Py_DecRef
_decref.argtypes = [ctypes.py_object]
_decref.restype = None

_incref = ctypes.pythonapi.Py_IncRef
_incref.argtypes = [ctypes.py_object]
_incref.restype = None


def patch_getitem(base_cls, func):
    assert callable(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        This wrapper returns the address of the resulting object as a
        python integer which is then converted to a pointer by ctypes
        """
        try:
            return func(*args, **kwargs)
        except NotImplementedError:
            return NotImplementedRet

    tp_as_name, impl_method = "tp_as_mapping", "mp_subscript"
    tyobj = PyTypeObject.from_address(id(base_cls))
    _incref(tyobj)
    struct_ty = PyMappingMethods
    tp_as_ptr = getattr(tyobj, tp_as_name)
    if not tp_as_ptr:
        # allocate new array
        tp_as_obj = struct_ty()
        tp_as_new_ptr = ctypes.cast(ctypes.addressof(tp_as_obj), ctypes.POINTER(struct_ty))

        setattr(tyobj, tp_as_name, tp_as_new_ptr)
    tp_as = tp_as_ptr[0]

    # find the C function type
    for fname, ftype in struct_ty._fields_:
        if fname == impl_method:
            cfunc_t = ftype

    cfunc = cfunc_t(wrapper)
    orig = ctypes.cast(getattr(tp_as, impl_method), ctypes.c_void_p)
    setattr(tp_as, impl_method, cfunc)
    return orig


def revert_getitem(base_cls, func):
    tp_as_name, impl_method = "tp_as_mapping", "mp_subscript"
    tyobj = PyTypeObject.from_address(id(base_cls))
    struct_ty = PyMappingMethods
    tp_as_ptr = getattr(tyobj, tp_as_name)
    tp_as = tp_as_ptr[0]

    # find the C function type
    for fname, ftype in struct_ty._fields_:
        if fname == impl_method:
            cfunc_t = ftype

    orig = ctypes.cast(func, cfunc_t)
    setattr(tp_as, impl_method, orig)
    _decref(tyobj)
