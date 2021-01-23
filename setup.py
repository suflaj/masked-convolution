# Copyright 2021 Miljenko Šuflaj
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="masked-convolution",
    version="0.1.0",
    description="A PyTorch wrapper for masked convolutions",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: GPU",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="pytorch torch convolution mask",
    url="http://github.com/suflaj/masked-convolution",
    author="Miljenko Šuflaj",
    author_email="headsouldev@gmail.com",
    license="Apache License 2.0",
    packages=["masked-convolution"],
    include_package_data=True,
    zip_safe=False,
)
