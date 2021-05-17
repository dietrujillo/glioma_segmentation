from typing import Tuple, AnyStr, Union

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
                 use_batch_norm: bool = True, activation: Union[Activation, AnyStr] = ReLU,
                 name: AnyStr = "", **kwargs):
        super(ConvBlock, self).__init__(name=name, **kwargs)
        self.conv = Conv3D(filters, kernel_size, padding="same", name=name+"_conv")
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn = BatchNormalization(name=name+"_bn")
        try:
            self.activation = activation(name=name + "_act")
        except TypeError:
            self.activation = Activation(activation, name=name + "_act")

    def call(self, inputs, **kwargs):
        conv_output = self.conv(inputs)
        if self.use_batch_norm:
            bn_outputs = self.bn(conv_output)
            activation_outputs = self.activation(bn_outputs)
        else:
            activation_outputs = self.activation(conv_output)
        return activation_outputs

    def build(self, input_shape):
        self.compute_output_shape(input_shape=input_shape)


def _calculate_concat_input(concat_1_shape, concat_2_shape):
    if len(concat_1_shape) != len(concat_2_shape) or concat_1_shape[1:-1] != concat_2_shape[1:-1]:
        raise ValueError(f"Shape mismatch in Concatenate layer: {concat_1_shape}, {concat_2_shape}")
    out = list(concat_1_shape)
    out[-1] += concat_2_shape[-1]
    return tuple(out)


class UNet(tf.keras.models.Model):
    """
    Basic U-Net architecture.
    """
    def __init__(self, **kwargs):
        super(UNet, self).__init__(**kwargs)

        self.e1_l1 = ConvBlock(32, (3, 3, 3), name="e1_l1")
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
        self.concatenate_3 = Concatenate(axis=-1, name="concatenate_3")

        self.d3_l1 = ConvBlock(256, (3, 3, 3), name="d3_l1")
        self.d3_l2 = ConvBlock(128, (3, 3, 3), name="d3_l2")

        self.d2_upsample = UpSampling3D(size=(2, 2, 2), name="d2_upsample")
        self.concatenate_2 = Concatenate(axis=-1, name="concatenate_2")

        self.d2_l1 = ConvBlock(128, (3, 3, 3), name="d2_l1")
        self.d2_l2 = ConvBlock(64, (3, 3, 3), name="d2_l2")

        self.d1_upsample = UpSampling3D(size=(2, 2, 2), name="d1_upsample")
        self.concatenate_1 = Concatenate(axis=-1, name="concatenate_1")

        self.d1_l1 = ConvBlock(64, (3, 3, 3), name="d3_l1")
        self.d1_l2 = ConvBlock(32, (3, 3, 3), name="d3_l2")

        self.out = ConvBlock(3, (1, 1, 1), use_batch_norm=False, activation="sigmoid", name="out")

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
        concat_1 = self.concatenate_1([e1_l2, d1_upsample])

        d1_l1 = self.d1_l1(concat_1)
        d1_l2 = self.d1_l2(d1_l1)

        out = self.out(d1_l2)

        return out

    def get_config(self):
        return super().get_config()

    def build(self, input_shape):

        super(UNet, self).build(input_shape=input_shape)

        self.e1_l1.build(input_shape=input_shape)
        e1_l1_output = self.e1_l1.compute_output_shape(input_shape=input_shape)
        self.e1_l2.build(input_shape=e1_l1_output)
        e1_l2_output = self.e1_l2.compute_output_shape(input_shape=e1_l1_output)

        self.e1_downsample.build(input_shape=e1_l2_output)
        e1_downsample_output = self.e1_downsample.compute_output_shape(input_shape=e1_l2_output)

        self.e2_l1.build(input_shape=e1_downsample_output)
        e2_l1_output = self.e2_l1.compute_output_shape(input_shape=e1_downsample_output)
        self.e2_l2.build(input_shape=e2_l1_output)
        e2_l2_output = self.e2_l2.compute_output_shape(input_shape=e2_l1_output)

        self.e2_downsample.build(input_shape=e2_l2_output)
        e2_downsample_output = self.e2_downsample.compute_output_shape(input_shape=e2_l2_output)

        self.e3_l1.build(input_shape=e2_downsample_output)
        e3_l1_output = self.e3_l1.compute_output_shape(input_shape=e2_downsample_output)
        self.e3_l2.build(input_shape=e3_l1_output)
        e3_l2_output = self.e3_l2.compute_output_shape(input_shape=e3_l1_output)

        self.e3_downsample.build(input_shape=e3_l2_output)
        e3_downsample_output = self.e3_downsample.compute_output_shape(input_shape=e3_l2_output)

        self.bottom_l1.build(input_shape=e3_downsample_output)
        bottom_l1_output = self.bottom_l1.compute_output_shape(input_shape=e3_downsample_output)
        self.bottom_l2.build(input_shape=bottom_l1_output)
        bottom_l2_output = self.bottom_l2.compute_output_shape(input_shape=bottom_l1_output)

        self.d3_upsample.build(input_shape=bottom_l2_output)
        d3_upsample_output = self.d3_upsample.compute_output_shape(input_shape=bottom_l2_output)

        concatenate_3_input = [e3_l2_output, d3_upsample_output]
        self.concatenate_3.build(input_shape=concatenate_3_input)
        concatenate_3_output = self.concatenate_3.compute_output_shape(input_shape=concatenate_3_input)

        self.d3_l1.build(input_shape=concatenate_3_output)
        d3_l1_output = self.d3_l1.compute_output_shape(input_shape=concatenate_3_output)
        self.d3_l2.build(input_shape=d3_l1_output)
        d3_l2_output = self.d3_l2.compute_output_shape(input_shape=d3_l1_output)

        self.d2_upsample.build(input_shape=d3_l2_output)
        d2_upsample_output = self.d2_upsample.compute_output_shape(input_shape=d3_l2_output)

        concatenate_2_input = [e2_l2_output, d2_upsample_output]
        self.concatenate_2.build(input_shape=concatenate_2_input)
        concatenate_2_output = self.concatenate_2.compute_output_shape(input_shape=concatenate_2_input)

        self.d2_l1.build(input_shape=concatenate_2_output)
        d2_l1_output = self.d2_l1.compute_output_shape(input_shape=concatenate_2_output)
        self.d2_l2.build(input_shape=d2_l1_output)
        d2_l2_output = self.d2_l2.compute_output_shape(input_shape=d2_l1_output)

        self.d1_upsample.build(input_shape=d2_l2_output)
        d1_upsample_output = self.d1_upsample.compute_output_shape(input_shape=d2_l2_output)

        concatenate_1_input = [e1_l2_output, d1_upsample_output]
        self.concatenate_1.build(input_shape=concatenate_1_input)
        concatenate_1_output = self.concatenate_1.compute_output_shape(input_shape=concatenate_1_input)

        self.d1_l1.build(input_shape=concatenate_1_output)
        d1_l1_output = self.d1_l1.compute_output_shape(input_shape=concatenate_1_output)
        self.d1_l2.build(input_shape=d1_l1_output)
        d1_l2_output = self.d1_l2.compute_output_shape(input_shape=d1_l1_output)

        self.out.build(input_shape=d1_l2_output)
