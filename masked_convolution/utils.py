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

from typing import Any

import torch
from torch import nn


def assert_type(x, x_name: str, wanted_type) -> Any:
    """Asserts that x, named x_name is of type wanted_type (or at least one of them).

    Args:
        x: A variable.
        x_name (str): The name of the variable.
        wanted_type: A type or a tuple of types.

    Raises:
        TypeError: Raised if x is not of wanted_type (or at least one of them).

    Returns:
        Any: The original x.
    """
    try:
        iter(wanted_type)
        type_string = (
            ", ".join(x.__name__ for x in wanted_type[:-1])
            + " or "
            + wanted_type[-1].__name__
        )
    except TypeError:
        type_string = str(wanted_type.__name__)

    if not isinstance(x, wanted_type):
        raise TypeError(
            f"Expected argument {x_name} to be {type_string}, but it is "
            f"{type(x).__name__}."
        )

    return x


def assert_mask_compatible_with_kernel(
    conv_weight: nn.Parameter, conv_weight_name: str, mask: torch.Tensor, mask_name: str
) -> torch.Tensor:
    """Asserts that mask, named mask_name has a compatible shape with conv_weight, named
    conv_weight_name, and corrects masks of shape (1, 1, *filter.shape) to filter.shape.

    Args:
        conv_weight (nn.Parameter): The convolution layer weight.
        conv_weight_name (str): The name of the convolution layer weight.
        mask (torch.Tensor): The mask tensor.
        mask_name (str): The name of the mask tensor.

    Raises:
        ValueError: Raised if mask kernel shape isn't identical to the conv_weight
        kernel shape.
        ValueError: Raised when the 2nd dimension of mask isn't identical to the 2nd
        dimension of conv_weight.
        ValueError: Raised when the 1st dimension of mask isn't identical to the 1st
        dimension of conv_weight.
        ValueError: Raised when mask is of different shape than conv_weight, but
        neither the 1st nor the 2nd dimensions are identical.
        ValueError: Raised if mask has a different number of dimensions than
        conv_weight, but it's not exactly 2 dimensions less.
        ValueError: Raised if mask shape isn't identical to the conv_weight kernel
        shape.

    Returns:
        torch.Tensor: The potentially corrected mask tensor.
    """
    conv_weight_shape = conv_weight.shape
    mask_shape = mask.shape

    if len(conv_weight_shape) == len(mask_shape):
        if conv_weight_shape != mask_shape:
            if conv_weight_shape[2:] != mask_shape[2:]:
                raise ValueError(
                    f"Expected {conv_weight_name} shape ({conv_weight_shape}) to have "
                    f"identical dimensions to {mask_name} shape ({mask_shape}) after "
                    "the first 2 dimensions (out_channels and in_channels)."
                )
            else:
                if mask_shape[0] == 1:
                    if mask_shape[1] == 1:
                        mask = mask.reshape(conv_weight_shape[2:])
                    elif mask_shape[1] != conv_weight_shape[1]:
                        raise ValueError(
                            f"Expected {mask_name} shape to have identical 2nd "
                            f"dimension as {conv_weight_name} shape "
                            f"({conv_weight_shape[1]}) when its 1st dimension is 1, "
                            f"but it is {mask_shape[1]} instead."
                        )
                elif mask_shape[1] == 1:
                    if mask_shape[0] != conv_weight_shape[0]:
                        raise ValueError(
                            f"Expected {mask_name} shape to have identical 1st "
                            f"dimension as {conv_weight_name} shape "
                            f"({conv_weight_shape[0]}) when its 2nd dimension is 1, "
                            f"but it is {mask_shape[0]} instead."
                        )
                else:
                    raise ValueError(
                        f"Expected {mask_name} shape ({mask_shape}) to have at least "
                        f"one of the first 2 dimension equal to 1 when its shape "
                        f"is not identical to {conv_weight_name} shape "
                        f"({conv_weight_shape})."
                    )

    # Reason this is not an else is because we modify the mask
    # so we don't want to skip checking if its shape is identical
    # to the filter shape.
    if len(conv_weight_shape) != mask_shape:
        if (len(conv_weight_shape) - 2) != len(mask_shape):
            raise ValueError(
                f"Expected {mask_name} shape to have exactly 2 dimensions less than "
                f"{conv_weight_name} shape ({len(conv_weight_shape) - 2}), but instead "
                f"it has {len(mask_shape)} dimensions."
            )

        if conv_weight_shape[2:] != mask_shape:
            raise ValueError(
                f"Expected {conv_weight_name} shape ({conv_weight_shape}) to have "
                f"identical dimensions to {mask_name} shape ({mask_shape}) after the "
                f"first 2 dimensions (out_channels and in_channels)."
            )

    return mask
