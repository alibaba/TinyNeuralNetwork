import importlib.util
import logging
import os
import sys
import types
import typing
from functools import wraps

loggers = []


class LazyExpression(object):
    """An expression object that can be lazily evaluated"""

    expr: typing.Callable[[], typing.Any]

    def __init__(self, expr: typing.Callable[[], typing.Any]):
        """Constructs a new LazyExpression object

        Args:
            expr (typing.Callable[[], typing.Any]): the expression
        """

        self.expr = expr

    def eval(self) -> typing.Any:
        """Evaluates the lazy expression

        Returns:
            typing.Any: the value of the expression
        """
        return self.expr()


class LazyObject(object):
    """An object that can be construct lazily using lazy expressions"""

    cls: type
    pos_options: list
    kw_options: typing.Dict[str, typing.Any]
    current_obj: typing.Any

    def __init__(
        self,
        cls: type,
        pos_options: typing.Optional[list] = None,
        kw_options: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ):
        """Constructs a new lazy object

        Args:
            cls (type): The type of the class to be construct lazily
            pos_options (typing.Optional[list], optional): The positional options. Defaults to None.
            kw_options (typing.Optional[typing.Dict[str, typing.Any]], optional): The keyword options. Defaults to None.
        """

        self.cls = cls
        if pos_options is None:
            self.pos_options = []
        else:
            self.pos_options = pos_options

        if kw_options is None:
            self.kw_options = {}
        else:
            self.kw_options = kw_options

        # Try construct object once
        self.get_next()

    def __str__(self) -> str:
        """Proxy __str__ to the underlying the object

        Returns:
            str: String representation of the underlying object
        """

        return str(self.current_obj)

    def __repr__(self) -> str:
        """Proxy __repr__ to the underlying the object

        Returns:
            str: String representation of the underlying object
        """

        return self.current_obj.__repr__()

    def get_current(self) -> typing.Any:
        """Gets the current copy of the lazily-constructed object

        Returns:
            typing.Any: The current copy of the lazily-constructed object
        """

        return self.current_obj

    def get_next(self):
        """Create a new object with the class and the arguments"""

        pos_options = []
        kw_options = {}
        for opt in self.pos_options:
            if isinstance(opt, LazyObject):
                pos_options.append(opt.get_current())
            elif isinstance(opt, LazyExpression):
                pos_options.append(opt.eval())
            else:
                pos_options.append(opt)

        for opt_k, opt_v in self.kw_options.items():
            if isinstance(opt_v, LazyObject):
                kw_options[opt_k] = opt.get_current()
            elif isinstance(opt_v, LazyExpression):
                kw_options[opt_k] = opt_v.eval()
            else:
                kw_options[opt_k] = opt_v

        self.current_obj = self.cls(*pos_options, **kw_options)
        return self.current_obj


def get_actual_type(param_type: type) -> typing.List[type]:
    """Gets the candidate actual types of a given type

    Args:
        param_type (type): The type to extract actual type from

    Returns:
        typing.List[type]: The candidate types
    """

    param_types = []
    if hasattr(param_type, '__origin__'):
        if param_type.__origin__ is typing.Union:
            for arg in param_type.__args__:
                param_types.extend(get_actual_type(arg))
        else:
            param_types.append(param_type.__origin__)
    else:
        param_types.append(param_type)
    return param_types


def conditional(cond: typing.Callable[[], bool]) -> typing.Callable:
    """A function wrapper that only runs the code of the function under given condition

    Args:
        cond (typing.Callable[[], bool]): The predicate given

    Returns:
        typing.Callable: A new function that only runs under given condition
    """

    def conditional_decorator(f):
        @wraps(f)
        def wrapper(*args, **kwds):
            if cond():
                return f(*args, **kwds)

        return wrapper

    return conditional_decorator


def class_conditional(cond: typing.Callable[[typing.Any], bool], default_val=None) -> typing.Callable:
    """A class function wrapper that only runs the code of the function under given condition

    Args:
        cond (typing.Callable[[], bool]): The predicate given

    Returns:
        typing.Callable: A new function that only runs under given condition
    """

    def conditional_decorator(f):
        @wraps(f)
        def wrapper(*args, **kwds):
            if cond(args[0]):
                return f(*args, **kwds)
            else:
                return default_val

        return wrapper

    return conditional_decorator


def tensors2ndarray(tensors) -> list:
    """Convert tensors in arbitrary format into list of ndarray

    Args:
        output: tensors in arbitrary format

    Returns:
        typing.List[numpy.ndarray]: list of ndarray
    """

    new_tensors = []
    if isinstance(tensors, (list, tuple)):
        for i in tensors:
            new_tensors.extend(tensors2ndarray(i))
    elif isinstance(tensors, dict):
        for k, v in tensors.items():
            new_tensors.extend(tensors2ndarray(v))
    elif hasattr(tensors, 'detach') and hasattr(tensors, 'numpy'):
        new_tensors.append(tensors.detach().numpy())
    elif not isinstance(tensors, (bool, str, int, float, types.FunctionType)):
        for k in tensors.__dir__():
            if not k.startswith('__'):
                v = getattr(tensors, v)
                new_tensors.extend(tensors2ndarray(v))
    return new_tensors


def import_from(module: str, name: str):
    """Import a module with the given name (Equivalent of `from module import name`)

    Args:
        module (str): The namespace of the module
        name (str): The name to be imported

    Returns:
        The imported class, function, and etc
    """

    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def import_from_path(module: str, path: str, name: str):
    """Import a module with the given name and path (Equivalent of `from module import name`)

    Args:
        module (str): The namespace of the module
        path (str): The path of the file
        name (str): The name to be imported

    Returns:
        The imported class, function, and etc
    """

    spec = importlib.util.spec_from_file_location(module, path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    sys.modules[module] = foo
    return getattr(foo, name)


def get_logger(name: str, level: typing.Optional[str] = None) -> logging.Logger:
    """Acquires the logger for a module with the given name

    Args:
        name (str): The name of the module
        level (typing.Optional[str]): The level of the logger. Defaults to None ('INFO').

    Returns:
        logging.Logger: The logger of the module
    """

    if level is None:
        level = 'INFO'

    level = os.environ.get('LOGLEVEL', level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Initialze the log level of the logger. Other possible values are `INFO`, `DEBUG` and `ERROR`
    logging.basicConfig(format='%(levelname)s (%(name)s) %(message)s')

    loggers.append(logger)

    return logger


def set_global_log_level(level: str = "INFO"):
    for logger in loggers:
        logger.setLevel(level)
