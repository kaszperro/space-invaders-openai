import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CnnNeuralNet(nn.Module):
    def __init__(self, input_image_dims, actions_space):
        """

        :param input_image_dims: (N, M, K) - num images, width, height
        :param actions_space:
        """
        super().__init__()

        self.input_image_dims = input_image_dims
        self.actions_space = actions_space

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(self.input_image_dims[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1)
        )

        self.linear_layer1 = nn.Linear(5376, 500)
        self.linear_layer2 = nn.Linear(500, self.actions_space)

    def forward(self, input_frame: Tensor):
        batches = input_frame.shape[0]
        cnn_out = self.cnn_layers(input_frame)
        cnn_out = F.relu(cnn_out)
        cnn_out = cnn_out.view(batches, -1)
        linear_out = self.linear_layer1(cnn_out)
        linear_out = F.relu(linear_out)
        return self.linear_layer2(linear_out)
