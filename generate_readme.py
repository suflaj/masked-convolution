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

import json
from textwrap import dedent

benchmark_key_to_string = {
    "conv1d": "Convolution 1D",
    "conv2d": "Convolution 2D",
    "conv3d": "Convolution 3D",
    "masked_conv1d": "Masked Convolution 1D",
    "masked_conv2d": "Masked Convolution 2D",
    "masked_conv3d": "Masked Convolution 3D",
}

benchmark_key_pairs = (
    ("conv1d", "masked_conv1d"),
    ("conv2d", "masked_conv2d"),
    ("conv3d", "masked_conv3d"),
)


def generate_benchmark_markdown() -> str:
    string = ""

    with open("data/benchmark.json") as f:
        json_dict = json.load(f)

    for first, second in benchmark_key_pairs:
        first_len = float(json_dict[first])
        second_len = float(json_dict[second])

        throughput_percentage = first_len / second_len * 100

        string += (
            f"- {benchmark_key_to_string[second]}: **{throughput_percentage:.02f} %** "
            f"{benchmark_key_to_string[first]} throughput\n        "
        )

    return string


def generate_readme_markdown() -> str:
    return dedent(
        f"""\
        # Masked Convolution

        [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

        A PyTorch implementation of a thin wrapper for masked convolutions.


        ## What are masked convolutions?

        Similarly to [partial convolutions](https://github.com/NVIDIA/partialconv), masked convolutions mask a part of the kernel, essentially ignoring data at specific locations. For an example, consider

        ```python
        a = [1, 2, 3, 4, 5]
        ```

        assuming we have a convolution kernel

        ```python
        kernel = [1, 1, 1]
        ```

        convolving over `a` would give us

        ```python
        a_conv = [6, 9, 12]
        ```

        However, if we were to mask the convolution kernel with a mask

        ```python
        mask = [1, 0, 1]
        ```

        **masked convolving** over `a` would return

        ```python
        a_masked_conv = [4, 6, 8]
        ```

        One use of masked convolutions is emulating skip-grams.


        ## Installation

        First, make sure you have PyTorch installed. This was tested on **Python 3.8** and **PyTorch 1.7.1**. Further testing is needed to determine whether it works on a different setup - chances are it does. The recommended way to install this is through PyPi by running:

        ```bash
        pip install masked-convolution
        ```

        Other than that, you can clone this repository, and in its root directory (where `setup.py` is located) run

        ```bash
        pip install .
        ```

        ## Benchmarks

        Every build, automatic benchmarks are run in order to determine how much overhead the implementation brings. The ordinary convolutions are used as a baseline, while the the performance of masked convolutions is described as a percentage of throughput of their respective baselines.

        Keep in mind that these benchmarks are in no way professional, they only serve to give users a general idea. Their results greatly differ, so they should be taken with a grain of salt.

        {generate_benchmark_markdown()}
        """
    )


def main():
    with open("README.md", mode="w+", encoding="utf8", errors="replace") as f:
        f.write(generate_readme_markdown())


if __name__ == "__main__":
    main()
