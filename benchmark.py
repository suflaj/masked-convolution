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

from datetime import datetime, timedelta
import json

import torch
from torch import nn

from masked_convolution import MaskedConvolution


def alternating_zero_one(length: int):
    return [i % 2 for i in range(length)]


def get_duration(data: torch.Tensor, layer: nn.Module) -> timedelta:
    start = datetime.now()

    for batch in data:
        layer(batch)

    end = datetime.now()

    return end - start


def main():
    print("Benchmarking...")

    # Shape is (n_samples, batch_size, in_channels, (feature_dims))
    data1d = torch.normal(0, 1, (20000, 32, 1, 5))
    data2d = torch.normal(0, 1, (10000, 32, 1, 5, 5))
    data3d = torch.normal(0, 1, (5000, 32, 1, 5, 5, 5))

    # All the same 1 channel in, 1 channel out, kernel size 3 convs
    conv1d = nn.Conv1d(1, 1, 3)
    conv2d = nn.Conv2d(1, 1, 3)
    conv3d = nn.Conv3d(1, 1, 3)

    # Alternating 0s and 1s as the kernel mask
    mask1d = torch.tensor(alternating_zero_one(3), dtype=torch.float).reshape(
        conv1d.weight.shape[2:]
    )
    mask2d = torch.tensor(alternating_zero_one(3 * 3), dtype=torch.float).reshape(
        conv2d.weight.shape[2:]
    )
    mask3d = torch.tensor(alternating_zero_one(3 * 3 * 3), dtype=torch.float).reshape(
        conv3d.weight.shape[2:]
    )

    masked_conv1d = MaskedConvolution(conv_layer=conv1d, mask=mask1d)
    masked_conv2d = MaskedConvolution(conv_layer=conv2d, mask=mask2d)
    masked_conv3d = MaskedConvolution(conv_layer=conv3d, mask=mask3d)

    conv_dict = {
        "conv1d": conv1d,
        "conv2d": conv2d,
        "conv3d": conv3d,
        "masked_conv1d": masked_conv1d,
        "masked_conv2d": masked_conv2d,
        "masked_conv3d": masked_conv3d,
    }

    data_list = [data1d, data2d, data3d, data1d, data2d, data3d]

    result_dict = {}

    for data, layer_name, layer in zip(data_list, *zip(*conv_dict.items())):
        result_dict[layer_name] = get_duration(data=data, layer=layer).microseconds

    with open("data/benchmark.json", mode="w+", encoding="utf8", errors="replace") as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
    main()
