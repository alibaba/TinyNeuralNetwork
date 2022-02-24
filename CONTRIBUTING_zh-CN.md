## 为项目做贡献
[English](CONTRIBUTING.md)

我们感谢你对我们的项目感兴趣，并且对于外部的贡献表示欢迎。

### 为项目贡献代码

#### 格式化

我们使用`flake8`和`black`来完成/检查代码的格式化，请使用下面的命令来安装和使用他们。
```bash
pip install black flake8

python3 -m black .
python3 -m flake8
```

或者你也可以使用`pre-commit`来自动在`git commit`前完成对应的检查。
```py
# 安装pre-commit以及我们配置好的git hook
pip install pre-commit
pre-commit install

# 在提交代码之前会自动完成检查
git commit -m "test"

# 或者也可以主动触发检查
pre-commit run
```

#### 函数注解和文档字符串

原则上添加的每个公共函数或者类方法都需要加上函数注解以及文档字符串。函数注解采用[typing](https://docs.python.org/3/library/typing.html)来标记。文档字符串采用[Google样式](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)。样例代码
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

#### 单元测试

对于模型转换器，每添加一个算子或者优化pass，需要添加相应的单元测试。
代码的位置：
- [算子](tests/converter_op_test.py)
- [优化pass](tests/converter_op_test.py)
