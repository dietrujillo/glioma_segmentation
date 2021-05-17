from models.unet import UNet, ConvBlock
import tensorflow as tf
from typing import AnyStr, Union
from tensorflow.keras.layers import (
    Activation, MaxPool3D, ReLU, Concatenate
)


class InceptionBlock(tf.keras.layers.Layer):
    """
    Inception module.
    """
    def __init__(self,
                 filters_1x1: int,
                 filters_3x3: int,
                 filters_5x5: int,
                 activation: Union[Activation, AnyStr] = ReLU,
                 name: AnyStr = "", **kwargs):
        super(InceptionBlock, self).__init__(name=name, **kwargs)

        self.conv1x1 = ConvBlock(filters=filters_1x1, kernel_size=(1, 1, 1),
                                 use_batch_norm=True, activation=activation, name=f"{name}_conv1x1")

        self.conv3x3_1 = ConvBlock(filters=filters_1x1, kernel_size=(1, 1, 1),
                                   use_batch_norm=True, activation=activation, name=f"{name}_conv3x3_1")
        self.conv3x3_2 = ConvBlock(filters=filters_3x3, kernel_size=(3, 3, 3),
                                   use_batch_norm=True, activation=activation, name=f"{name}_conv3x3_2")

        self.conv5x5_1 = ConvBlock(filters=filters_1x1, kernel_size=(1, 1, 1),
                                   use_batch_norm=True, activation=activation, name=f"{name}_conv5x5_1")
        self.conv5x5_2 = ConvBlock(filters=filters_5x5, kernel_size=(5, 5, 5),
                                   use_batch_norm=True, activation=activation, name=f"{name}_conv5x5_2")

        self.pooling_1 = MaxPool3D(pool_size=(3, 3, 3), padding="same", name=f"{name}_pooling_1")
        self.pooling_2 = ConvBlock(filters=filters_1x1, kernel_size=(1, 1, 1),
                                   use_batch_norm=True, activation=activation, name=f"{name}_pooling_2")

        self.concat = Concatenate(axis=-1, name=f"{name}_concat")

    def call(self, inputs, **kwargs):
        conv_1x1_output = self.conv1x1(inputs)

        conv_3x3_output = self.conv3x3_1(inputs)
        conv_3x3_output = self.conv3x3_2(conv_3x3_output)

        conv_5x5_output = self.conv5x5_1(inputs)
        conv_5x5_output = self.conv5x5_2(conv_5x5_output)

        pooling_output = self.pooling_1(inputs)
        pooling_output = self.pooling_2(pooling_output)

        out = self.concat([conv_1x1_output, conv_3x3_output, conv_5x5_output, pooling_output])

        return out


class InceptionUNet(UNet):
    """
    U-Net architecture with inception blocks at every convolutional step save for the last one.
    """
    def __init__(self, **kwargs):
        super(InceptionUNet, self).__init__(**kwargs)

        self.e1_l1 = InceptionBlock(16, 16, 16, name="e1_l1")
        self.e1_l2 = InceptionBlock(32, 32, 32, name="e1_l2")

        self.e2_l1 = InceptionBlock(32, 32, 32, name="e2_l1")
        self.e2_l2 = InceptionBlock(64, 64, 64, name="e2_l2")

        self.e3_l1 = InceptionBlock(64, 64, 64, name="e3_l1")
        self.e3_l2 = InceptionBlock(128, 128, 128, name="e3_l2")

        self.bottom_l1 = InceptionBlock(128, 128, 128, name="bottom_l1")
        self.bottom_l2 = InceptionBlock(256, 256, 256, name="bottom_l2")

        self.d3_l1 = InceptionBlock(128, 128, 128, name="d3_l1")
        self.d3_l2 = InceptionBlock(64, 64, 64, name="d3_l2")

        self.d2_l1 = InceptionBlock(64, 64, 64, name="d2_l1")
        self.d2_l2 = InceptionBlock(32, 32, 32, name="d2_l2")

        self.d1_l1 = InceptionBlock(32, 32, 32, name="d3_l1")
        self.d1_l2 = InceptionBlock(16, 16, 16, name="d3_l2")
