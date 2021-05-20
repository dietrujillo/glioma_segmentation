from typing import Union, AnyStr, Any, List, Dict

import tensorflow as tf

from models.unet import ConvBlock


class ModelAggregator(tf.keras.models.Model):
    """
    Aggregator class. Will combine different models and aggregate their predictions.
    """
    def __init__(self,
                 models: List[Union[type, AnyStr]],
                 pretrained: bool = True,
                 model_params: List[Dict[AnyStr, Any]] = None,
                 custom_layers: List[Dict[AnyStr, Any]] = None,
                 output_channels: int = 3,
                 use_inputs: bool = True,
                 **kwargs):
        """
        Initializes model aggregator.
        :param models: list of models. Can be model classes or str paths to saved models if pretrained=True.
        :param pretrained: whether models to aggregate have been trained before aggregating or
        will be trained along with the aggregator.
        :param model_params: list of parameters for model constructors. Used only if pretrained=False.
        :param custom_layers: list of custom layers to use for model loading. Used only if pretrained=True.
        :param output_channels: number of expected channels in output for every model.
        :param use_inputs: whether to include the original inputs in the aggregation.
        :param kwargs: kwargs for base class constructor.
        """
        super(ModelAggregator, self).__init__(**kwargs)
        self.pretrained = pretrained
        self.models = []
        self.output_channels = output_channels

        self.use_inputs = use_inputs
        if self.use_inputs:
            self.input_conv = ConvBlock(filters=output_channels, kernel_size=(1, 1, 1), name="input_conv")

        self.channel_conv = [ConvBlock(filters=1, kernel_size=(1, 1, 1)) for _ in range(output_channels)]

        for i, model in enumerate(models):
            if pretrained:
                sub_model = tf.keras.models.load_model(model, custom_objects=custom_layers[i])
                sub_model.trainable = False
                self.models.append(sub_model)
            else:
                self.models.append(models[i](**model_params[i]))

    def call(self, inputs, training=None, mask=None):
        model_predictions = []
        if self.use_inputs:
            model_predictions.append(self.conv(inputs))

        for model in self.models:
            model_predictions.append(model(inputs, training=(training and not self.pretrained), mask=mask))

        assert all([x.shape[-1] == self.output_channels for x in model_predictions])
        model_predictions = tf.stack(model_predictions, axis=-2)

        ret = []
        for channel in range(self.output_channels):
            channel_prediction = self.channel_conv[channel](model_predictions[..., channel])
            ret.append(tf.squeeze(channel_prediction, axis=range(1, len(channel_prediction.shape))))

        return tf.stack(ret, axis=-1)

    def get_config(self):
        return super(ModelAggregator, self).get_config()
