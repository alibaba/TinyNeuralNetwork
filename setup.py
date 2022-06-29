import os
import setuptools
import subprocess

from datetime import datetime

try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements


def get_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_datetime():
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S")


predefined_version = os.getenv('TINYML_BUILD_VERSION')
if predefined_version:
    version = predefined_version
else:
    version = '0.1.0'
    sha = get_sha()
    dt = get_datetime()
    if sha:
        version += f'.{dt}+{sha}'
    else:
        version += f'.{dt}'

reqs = parse_requirements('requirements.txt', session=False)

install_reqs = [str(ir.req) if hasattr(ir, 'req') else str(ir.requirement) for ir in reqs]

setuptools.setup(
    name="TinyNeuralNetwork",
    version=version,
    author="Huanghao Ding, Jiachen Pu",
    description="A collection of tools that aims for the inference performance of a AI model in AIoT scenarios.",
    url="https://github.com/alibaba/TinyNeuralNetwork",
    project_urls={
        "Bug Tracker": "https://github.com/alibaba/TinyNeuralNetwork/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    package_data={'tinynn': ['graph/configs/*.yml']},
    python_requires=">=3.6",
    install_requires=install_reqs,
)
