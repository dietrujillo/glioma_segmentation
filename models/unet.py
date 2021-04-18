from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv3D, BatchNormalization, ReLU, MaxPool3D,
    Concatenate, UpSampling3D, Activation
)


class ConvBlock(tf.keras.layers.Layer):
    """
    Convolutional layer, with batch normalization and an activation function.
    """
    def __init__(self, filters: int, kernel_size: Tuple[int, int, int],
                 use_batch_norm: bool = True, activation: Activation = ReLU, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.conv = Conv3D(filters, kernel_size, padding="same")
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn = BatchNormalization()
        self.activation = activation()

    def call(self, inputs, **kwargs):
        conv_output = self.conv(inputs)
        if self.use_batch_norm:
            bn_outputs = self.bn(conv_output)
            activation_outputs = self.activation(bn_outputs)
        else:
            activation_outputs = self.activation(conv_output)
        return activation_outputs


class UNet(tf.keras.models.Model):
    """
    Basic U-Net model.
    """

    def __init__(self, input_shape, **kwargs):
        super(UNet, self).__init__(**kwargs)

        self.e1_l1 = ConvBlock(32, (3, 3, 3), name="e1_l1", input_shape=input_shape)
        self.e1_l2 = ConvBlock(64, (3, 3, 3), name="e1_l2")

        self.e1_downsample = MaxPool3D(pool_size=(2, 2, 2), name="max_pool_1")

        self.e2_l1 = ConvBlock(64, (3, 3, 3), name="e2_l1")
        self.e2_l2 = ConvBlock(128, (3, 3, 3), name="e2_l2")

        self.e2_downsample = MaxPool3D(pool_size=(2, 2, 2), name="max_pool_2")

        self.e3_l1 = ConvBlock(128, (3, 3, 3), name="e3_l1")
        self.e3_l2 = ConvBlock(256, (3, 3, 3), name="e3_l2")

        self.e3_downsample = MaxPool3D(pool_size=(2, 2, 2), name="max_pool_3")

        self.bottom_l1 = ConvBlock(256, (3, 3, 3), name="bottom_l1")
        self.bottom_l2 = ConvBlock(512, (3, 3, 3), name="bottom_l2")

        self.d3_upsample = UpSampling3D(size=(2, 2, 2), name="d3_upsample")
        self.concatenate_3 = Concatenate(axis=-1, name="concat_3")

        self.d3_l1 = ConvBlock(256, (3, 3, 3), name="d3_l1")
        self.d3_l2 = ConvBlock(128, (3, 3, 3), name="d3_l2")

        self.d2_upsample = UpSampling3D(size=(2, 2, 2), name="d2_upsample")
        self.concatenate_2 = Concatenate(axis=-1, name="concat_2")

        self.d2_l1 = ConvBlock(128, (3, 3, 3), name="d2_l1")
        self.d2_l2 = ConvBlock(64, (3, 3, 3), name="d2_l2")

        self.d1_upsample = UpSampling3D(size=(2, 2, 2), name="d1_upsample")
        self.concatenate_1 = Concatenate(axis=-1, name="concat_1")

        self.d1_l1 = ConvBlock(64, (3, 3, 3), name="d3_l1")
        self.d1_l2 = ConvBlock(32, (3, 3, 3), name="d3_l2")

        self.out = ConvBlock(4, (1, 1, 1), name="out")

        self.build(input_shape=input_shape)

    def call(self, inputs, training=None, mask=None):

        e1_l1 = self.e1_l1(inputs)
        e1_l2 = self.e1_l2(e1_l1)

        e1_downsample = self.e1_downsample(e1_l2)

        e2_l1 = self.e2_l1(e1_downsample)
        e2_l2 = self.e2_l2(e2_l1)

        e2_downsample = self.e2_downsample(e2_l2)

        e3_l1 = self.e3_l1(e2_downsample)
        e3_l2 = self.e3_l2(e3_l1)

        e3_downsample = self.e3_downsample(e3_l2)

        bottom_l1 = self.bottom_l1(e3_downsample)
        bottom_l2 = self.bottom_l2(bottom_l1)

        d3_upsample = self.d3_upsample(bottom_l2)
        concat_3 = self.concatenate_3([e3_l2, d3_upsample])

        d3_l1 = self.d3_l1(concat_3)
        d3_l2 = self.d3_l2(d3_l1)

        d2_upsample = self.d2_upsample(d3_l2)
        concat_2 = self.concatenate_2([e2_l2, d2_upsample])

        d2_l1 = self.d2_l1(concat_2)
        d2_l2 = self.d2_l2(d2_l1)

        d1_upsample = self.d1_upsample(d2_l2)
        concat_1 = self.concatenate_2([e1_l2, d1_upsample])

        d1_l1 = self.d1_l1(concat_1)
        d1_l2 = self.d1_l2(d1_l1)

        out = self.out(d1_l2)

        return out

    def get_config(self):
        return super().get_config()
