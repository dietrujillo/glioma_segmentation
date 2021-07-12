from typing import Union, AnyStr, Any, List, Dict

import tensorflow as tf
from tensorflow.keras.layers import PReLU

from models.unet import ConvBlock


class ModelAggregator(tf.keras.models.Model):
    """
    Aggregator class. Will combine different models and aggregate their predictions.
    """
    def __init__(self,
                 models: List[Union[type, AnyStr]],
                 custom_layers: List[Dict[AnyStr, Any]] = None,
                 output_channels: int = 3,
                 use_inputs: bool = True,
                 **kwargs):
        """
        Initializes model aggregator.
        :param models: list of paths to saved models.
        :param custom_layers: list of custom layers to use for model loading.
        :param output_channels: number of expected channels in output for every model.
        :param use_inputs: whether to include the original inputs in the aggregation.
        :param kwargs: kwargs for base class constructor.
        """
        super(ModelAggregator, self).__init__(**kwargs)
        self.models = []
        self.output_channels = output_channels

        self.use_inputs = use_inputs
        if self.use_inputs:
            self.input_conv_1 = ConvBlock(filters=32, kernel_size=(3, 3, 3), name="input_conv_1")
            self.input_conv_2 = ConvBlock(filters=self.output_channels, kernel_size=(1, 1, 1), activation="sigmoid", name="input_conv_2")

        self.channel_conv_0 = [ConvBlock(filters=32, kernel_size=(3, 3, 3), name=f"channel_conv_0_{i}") for i in range(output_channels)]
        self.channel_conv_1 = [ConvBlock(filters=32, kernel_size=(3, 3, 3), name=f"channel_conv_1_{i}") for i in range(output_channels)]
        self.channel_conv_2 = [ConvBlock(filters=8, kernel_size=(3, 3, 3), name=f"channel_conv_2_{i}") for i in range(output_channels)]
        self.channel_conv_3 = []
        for i in range(output_channels):
            self.channel_conv_3.append(ConvBlock(filters=1, kernel_size=(1, 1, 1), use_batch_norm=False, activation="sigmoid", name=f"channel_conv_2_{i}"))

        for i, model in enumerate(models):
            sub_model = tf.keras.models.load_model(model, custom_objects=custom_layers[i])
            sub_model.trainable = False
            self.models.append(sub_model)

    def call(self, inputs, training=None, mask=None):
        model_predictions = []
        if self.use_inputs:
            input_prediction = self.input_conv_1(inputs)
            input_prediction = self.input_conv_2(input_prediction)
            model_predictions.append(input_prediction)

        for model in self.models:
            model_predictions.append(model(inputs))

        assert all([x.shape[-1] == self.output_channels for x in model_predictions])
        model_predictions = tf.stack(model_predictions, axis=-2)

        ret = []
        for channel in range(self.output_channels):
            channel_prediction = self.channel_conv_0[channel](model_predictions[..., channel])
            channel_prediction = self.channel_conv_1[channel](channel_prediction)
            channel_prediction = self.channel_conv_2[channel](channel_prediction)
            channel_prediction = self.channel_conv_3[channel](channel_prediction)
            ret.append(channel_prediction)

        ret = tf.squeeze(tf.stack(ret, axis=-1), [-2])
        return ret

    def get_config(self):
        return super(ModelAggregator, self).get_config()

    def build(self, input_shape):
        super(ModelAggregator, self).build(input_shape=input_shape)
        if self.use_inputs:
            self.input_conv_1.build(input_shape=input_shape)
            self.input_conv_2.build(input_shape=(None, *input_shape[:-1], 32))
        for conv in self.channel_conv_0:
            conv.build(input_shape=(None, *input_shape[:-1], len(self.models) + (1 if self.use_inputs else 0)))
        for conv in self.channel_conv_1:
            conv.build(input_shape=(None, *input_shape[:-1], 32))
        for conv in self.channel_conv_2:
            conv.build(input_shape=(None, *input_shape[:-1], self.output_channels))