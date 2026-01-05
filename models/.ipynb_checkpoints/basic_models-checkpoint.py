import torch
from torch import nn
from torchvision import models


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) with customizable hidden layers.
    
    Args:
        - input_dim (int): The input dimension to the first fully connected layer (default: 512).
        - hidden_layers (list of int): List specifying the number of units in each hidden layer.
        
    """
    def __init__(self, input_dim: int=512, hidden_layers: list=[512,512,128]):
        super().__init__()
        self.out_features = hidden_layers[-1]
        self.projection_layers = nn.ModuleList()
        
        self.projection_layers.append(nn.Linear(input_dim, hidden_layers[0]))
        
        for i in range(1,len(hidden_layers)):
            self.projection_layers.append(nn.ReLU(inplace=True))
            self.projection_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        
    def forward(self, x):
        """
        Forward pass of the MLP module.
        
        Args:
            - x (Pytorch Tensor): Input tensor.
        
        Returns:
            - Output tensor after passing through the MLP layers.
            
        """
        for i in range(len(self.projection_layers)):
            x = self.projection_layers[i](x)
            
        return x  

class ConvBlock(nn.Module):
    """
    A custom convolutional block with Pytorch that consists of two convolution layers.
    
    Args:
        - in_channels (int): Number of channels in the input.
        - out_channels (int): Number of channels produced by the convolution.
        - kernel_size (int): Size of the convolving kernel (default: 3).
        - stride (int): Stride of the convolution (default: 1).
        - padding (int): Padding added to all four sides of the input (default: 1).
        
    """

    def __init__(self, in_channels: int=None, out_channels: int=None,
                 kernel_size: int=3, stride: int=1, padding: int=1):
        super().__init__()
        # Optional BatchNorm layers are commented out when training with few points (~10,000).
        
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        
        #self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        #self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Defines the forward pass through the convolutional block.
        
        Args:
            - x (Pytroch Tensor): Input image.
        
        Returns:
            - Pytorch Tensor with out_channels channels.
            
        """
        input_x = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.relu(out)

        return out
    
class JointBlocks(nn.Module):
    """
    A PyTorch module that combines multiple convolutional blocks and average pooling layers.
    
    Args:
        - input_channels (int): Number of input channels for the first convolutional block (default: 32).
        - block_channels (list of int): List specifying the output channels for each convolutional block.
        - avg_pooling_layers (list of int): List specifying the kernel size for each average pooling layer.
        
    """
    def __init__(self, input_channels: int=32, block_channels: list=[32,64,128], avg_pooling_layers: list=[2,2,4]):
        super().__init__()
        # Combines CNN blocks
        self.layers = nn.ModuleList()
        self.layers.append(ConvBlock(
            input_channels,
            block_channels[0]
        ))
        self.layers.append(nn.AvgPool2d(avg_pooling_layers[0]))
        
        for i in range(1,len(block_channels)):
            self.layers.append(ConvBlock(
                block_channels[i-1],
                block_channels[i]
            ))
            self.layers.append(nn.AvgPool2d(avg_pooling_layers[i]))
            
        # Flatten output.
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        """
        Defines the forward pass through the joint blocks.
        
        Args:
            - x (Pytorch Tensor): Input tensor.
        
        Returns:
            - Flattened output tensor after passing through all layers.
            
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = self.flatten(x)
        
        return x

class Encoder(nn.Module):
    """
    A PyTorch encoder module that applies an initial convolution layer followed by joint blocks.
    
    Args:
        - input_channels (int): Number of input channels (default: 4).
        - first_layer_output_channels (int): Number of output channels after the first convolution (default: 32).
        - joint_blocks (nn.Module): A module containing several convolutional blocks.
        
    """
        
    def __init__(self, input_channels: int=4, first_layer_output_channels: int=32, joint_blocks: nn.Module=None):
        super().__init__()
        # Optional BatchNorm layers are commented out when training with few points (~10,000).

        #self.bn_input = nn.BatchNorm2d(input_channels)
        self.conv1 = nn.Conv2d(
            input_channels,
            first_layer_output_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        #self.bn = nn.BatchNorm2d(first_layer_output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.joint_blocks = joint_blocks
        
    def forward(self, x):
        """
        Forward pass of the encoder module.
        
        Args:
            - x (Pytorch Tensor): Input image.
        
        Returns:
            - Pytorch Tensor: Flattened output of the encoder.
            
        """
        #x = self.bn_input(x)
        x = self.conv1(x)
        #x = self.bn(x)
        x = self.relu(x)
        x = self.joint_blocks(x)
        
        return x

class CustomConvNeXt(nn.Module):
    def __init__(self, n_filters):
        super(CustomConvNeXt, self).__init__()
        # Load pre-existing model (ConvNeXt)
        self.model = models.convnext_tiny(weights=None)
        
        # Replace the first convolutional layer
        self.model.features[0][0] = nn.Conv2d(n_filters, 96, kernel_size=(4, 4), stride=(4, 4))
    
    def forward(self, x):
        return self.model(x)

