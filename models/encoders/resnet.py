import torch
from torch import nn
from torch import Tensor

from collections import OrderedDict
from typing import Type, List


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        down_sample: nn.Module = None,
        activation: nn.modules.activation = nn.ReLU(inplace=True)
    ) -> None:
        super(BottleneckBlock, self).__init__()

        self.expansion = expansion
        self.activation = activation
        self.down_sample = down_sample

        # architecture

        self.block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            )),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('act1', activation),
            ('conv2', nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False
            )),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('act2', activation),
            ('conv3', nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * expansion,
                kernel_size=1,
                bias=False
            )),
            ('bn3', nn.BatchNorm2d(out_channels * expansion))
        ]))

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.block(inputs)

        residual = inputs
        if self.down_sample:
            residual = self.down_sample(residual)

        outputs = self.activation(outputs + residual)

        return outputs


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BottleneckBlock] = BottleneckBlock,
        layer_blocks: List[int] = [3, 4, 6, 3],
        num_classes: int = 512,
        expansion: int = 4,
    ) -> None:
        super(ResNet, self).__init__()

        self.current_channels = 64
        self.expansion = expansion
        self.down_sample = 2

        # architecture

        self.conv_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            ),
        )

        self.conv_1 = self.__make_layer(
            block=block,
            num_of_blocks=layer_blocks[0],
            in_channels=64,
        )

        self.conv_2 = self.__make_layer(
            block=block,
            num_of_blocks=layer_blocks[1],
            in_channels=128,
            stride=2
        )

        self.conv_3 = self.__make_layer(
            block=block,
            num_of_blocks=layer_blocks[2],
            in_channels=256,
            stride=2
        )

        self.conv_4 = self.__make_layer(
            block=block,
            num_of_blocks=layer_blocks[3],
            in_channels=512,
            stride=2
        )

        self.flatten = nn.Sequential(
            nn.Flatten(2),
            Permute((0, 2, 1)),
            nn.Linear(512 * self.expansion, num_classes)
        )

        self.init_parameters()

    def init_parameters(self):
        for i in range(5):
            block = getattr(self, f'conv_{i}')
            for name, param in block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def forward(self, inputs: Tensor) -> Tensor:
        partial_result = inputs
        for i in range(5):
            partial_result = getattr(self, f'conv_{i}')(partial_result)
        return self.flatten(partial_result)

    def __make_layer(
        self,
        block: Type[BottleneckBlock],
        num_of_blocks: int,
        in_channels: int,
        stride: int = 1
    ) -> nn.Sequential:
        """
        Make a layer with given block module
        :param block: block module to be used
        :param num_of_blocks: number of block layers
        :param in_channels: number od block input channels
        :param stride: stride
        """

        down_sample = None
        out_channels = in_channels * self.expansion
        if stride != 1 or self.current_channels != out_channels:
            self.down_sample *= stride
            down_sample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.current_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels)
            )

        layers = [
            block(
                in_channels=self.current_channels,
                out_channels=in_channels,
                stride=stride,
                down_sample=down_sample
            )
        ]

        # update and it will be used for next block layer
        self.current_channels = out_channels

        # in_channels:  in order to connect with prev block,
        #               input channel equals out channels
        # out_channels: block module has expansion,
        #               output channel equals input channels

        layers += [
            block(
                in_channels=out_channels,
                out_channels=in_channels,
            )
            for _ in range(1, num_of_blocks)
        ]

        return nn.Sequential(*layers)


class Permute(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(*self.dims)
