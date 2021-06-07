from typing import Union, AnyStr, Any, List, Dict

import tensorflow as tf

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
            self.input_conv = ConvBlock(filters=output_channels, kernel_size=(1, 1, 1), name="input_conv")

        self.channel_conv = [ConvBlock(filters=1, kernel_size=(1, 1, 1), activation="sigmoid", name=f"channel_conv_{i}") for i in range(output_channels)]

        for i, model in enumerate(models):
            sub_model = tf.keras.models.load_model(model, custom_objects=custom_layers[i])
            sub_model.trainable = False
            self.models.append(sub_model)

    def call(self, inputs, training=None, mask=None):
        model_predictions = []
        if self.use_inputs:
            model_predictions.append(self.input_conv(inputs))

        for model in self.models:
            model_predictions.append(model(inputs))

        assert all([x.shape[-1] == self.output_channels for x in model_predictions])
        model_predictions = tf.stack(model_predictions, axis=-2)

        ret = []
        for channel in range(self.output_channels):
            channel_prediction = self.channel_conv[channel](model_predictions[..., channel])
            ret.append(channel_prediction)

        ret = tf.squeeze(tf.stack(ret, axis=-1), [-2])
        return ret

    def get_config(self):
        return super(ModelAggregator, self).get_config()

    def build(self, input_shape):
        super(ModelAggregator, self).build(input_shape=input_shape)
        self.input_conv.build(input_shape=input_shape)
        for conv in self.channel_conv:
            conv.build(input_shape=(None, *input_shape[:-1], len(self.models)))