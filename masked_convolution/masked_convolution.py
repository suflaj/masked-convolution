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

import copy
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from . import utils


class MaskedConvolution(nn.Module):
    _mask_key = "mask"
    _allowed_classes = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

    @staticmethod
    def _check_init_arguments(
        conv_layer: nn.Module, mask: Optional
    ) -> Tuple[nn.Module, torch.Tensor]:
        """Checks MaskedConvolution.__init__() arguments, corrects them, and returns
        them corrected.

        Args:
            conv_layer (nn.Module): A convolutional layer you wish to mask. Must be an
            instance from one of the following classes:
            - torch.nn.Conv1d
            - torch.nn.Conv2d
            - torch.nn.Conv3d

            mask (Optional): A mask you wish to apply to the convolution kernel. If
            None, will create a mask of ones (ordinary convolution behaviour). Mask must
            be one of the following shapes:
            - (out_channels, in_channels, *kernel.shape)
            - (1, in_channels, *kernel.shape)
            - (out_channels, 1, *kernel.shape)
            - (1, 1, *kernel.shape)
            - kernel.shape

        Returns:
            Tuple[nn.Module, torch.Tensor]: The original conv_layer as well as the
            potentially corrected mask as a torch.Tensor.
        """
        utils.assert_type(conv_layer, "conv_layer", MaskedConvolution._allowed_classes)

        if mask is None:
            mask = torch.ones(size=conv_layer.weight.shape[2:])

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=conv_layer.weight.dtype)

        if not mask.dtype == conv_layer.weight.dtype:
            mask.type(conv_layer.weight.dtype)

        mask = utils.assert_mask_compatible_with_kernel(
            conv_weight=conv_layer.weight,
            conv_weight_name="conv_weight",
            mask=mask,
            mask_name="mask",
        )

        return conv_layer, mask

    @staticmethod
    def _parse_config(config: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Parses a MaskedConvolution config.

        Args:
            config (Dict[str, Any]): A dictionary that maps members to their values.

        Returns:
            Optional[torch.Tensor]: The mask as a torch.Tensor, or None if it's not
            present in the config.
        """
        mask = config.get(MaskedConvolution._mask_key)

        if mask is not None:
            mask = torch.tensor(mask)

        return mask

    def __init__(
        self,
        conv_layer: nn.Module = None,
        mask: Optional = None,
        check_arguments: bool = True,
        config: Dict[str, Any] = None,
    ):
        """The MaskedConvolution constructor.

        Args:
            conv_layer (nn.Module, optional): The convolutional layer you wish to mask.
            Defaults to None.
            mask (Optional, optional): The mask you wish to apply to the convolution
            layer kernel. Defaults to None (no masking).
            check_arguments (bool, optional): Determines whether arguments are checked
            at instantiation. Defaults to True.
            config (Dict[str, Any], optional): A config mapping member names to values.
            Overrides arguments. Defaults to None (not evaluated).
        """
        super().__init__()

        if config is not None:
            mask = self._parse_config(config=config)

        if check_arguments:
            conv_layer, mask = self._check_init_arguments(
                conv_layer=conv_layer, mask=mask
            )

        self._conv_layer = copy.deepcopy(conv_layer)
        self._mask = mask.detach().clone()

        with torch.no_grad():
            self._mask.clamp(min=0.0, max=1.0)
            torch.round(self._mask)

            self._conv_layer.weight = torch.nn.Parameter(
                self.mask * self.conv_layer.weight
            )

        self._conv_layer.weight.register_hook(lambda x: self.mask * x)

    # region Properties
    @property
    def conv_layer(self) -> nn.modules.conv._ConvNd:
        """The conv_layer property.

        Returns:
            nn.modules.conv._ConvNd: The masked convolution layer.
        """
        return self._conv_layer

    @property
    def mask(self) -> torch.Tensor:
        """The mask property.

        Returns:
            torch.Tensor: A torch.Tensor used to mask the convolution layer kernel.
        """
        return self._mask

    @property
    def config(self) -> Dict[str, Any]:
        """The config property.

        Returns:
            Dict[str, Any]: A config generated from the instance's members.
        """
        return {
            self._mask_key: self.mask.tolist(),
        }

    # endregion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward method.

        Args:
            x (torch.Tensor): A torch.Tensor that is passed as input to the masked
            convolution.

        Returns:
            torch.Tensor: The result of the masked convolution over x.
        """
        return self.conv_layer(x)
