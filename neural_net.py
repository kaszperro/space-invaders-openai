import torch.nn as nn
from torch import Tensor


class CnnNeuralNet(nn.Module):
    def __init__(self, input_image_dims, actions_space, num_cnn_layers):
        """

        :param input_image_dims: (N, M, K) - num images, width, height
        :param actions_space:
        """
        super().__init__()

        self.input_image_dims = input_image_dims
        self.actions_space = actions_space
        self.num_cnn_layers = num_cnn_layers

        prev_channels = self.input_image_dims[0]
        cnn_layers = []
        for i in range(self.num_cnn_layers):
            cnn_layers.append(nn.Conv2d(prev_channels, 32, 3))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d(2))
            prev_channels = 32

        self.cnn_layers = nn.Sequential(*cnn_layers)
        self.linear_layer = nn.Linear(32 * 104 * 79, self.actions_space)

    def forward(self, input_frame: Tensor):
        batches = input_frame.shape[0]
        cnn_out = self.cnn_layers(input_frame)
        cnn_out = cnn_out.view(batches, -1)
        return self.linear_layer(cnn_out)
