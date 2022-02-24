## Contributing to the project
[中文](CONTRIBUTING_zh-CN.md)

We appreciate your interest in our project and welcome external contributions.

### Contributing code to the project

#### Formatting

We use `flake8` and `black` to perform and check code formatting, please use the following commands to install and use them.
```bash
pip install black flake8

python3 -m black .
python3 -m flake8
```

Or you can use `pre-commit`, which performs the corresponding checks before `git commit` automatically.
```py
# Install pre-commit and pre-configured git hooks
pip install pre-commit
pre-commit install

# The code check will be done automatically before committing the code
git commit -m "test"

# Or you can also trigger the code check manually
pre-commit run
```

#### Function annotations and docstrings

Generally speaking, every public function or class method you add requires a function annotation and a docstring. Function annotations are marked with [typing](https://docs.python.org/3/library/typing.html). Docstrings are in [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). Sample code is show below.
```py
def test_func(param1: int, param2: str) -> bool:
    """Documentation for the function

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    """
```

#### Unit tests

As for the model converter, when an operator or optimization pass is added, the corresponding unit tests need to be written as well.
The location of the code for unit tests:
- [Operators](../tests/converter_op_test.py)
- [Optimization passes](../tests/converter_op_test.py)
