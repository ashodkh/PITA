import torch
from torch import nn
from torchvision import models
import monotonicnetworks as lmn

class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) with customizable hidden layers.
    
    Attributes:
        - input_dim (int): The input dimension to the first fully connected layer (default: 512).
        - hidden_layers (list of int): List specifying the number of units in each hidden layer.
        - output_dim (int): Output dimensions.
    """
    def __init__(self, input_dim: int=512, hidden_layers: list=[512,512,128], output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.projection_layers = nn.ModuleList()
        
        self.projection_layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.projection_layers.append(nn.ReLU(inplace=True))
        
        for i in range(1,len(hidden_layers)):
            self.projection_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.projection_layers.append(nn.ReLU(inplace=True))

        self.projection_layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
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

class LipschitzMLP(nn.Module):
    '''
    A multi-layer perceptron (MLP) with customizable hidden layers that is Lipschitz bounded with p=1.
    
    Attributes:
        - input_dim (int): The input dimension to the first fully connected layer (default: 512).
        - hidden_layers (list of int): List specifying the number of units in each hidden layer.
        - output_dim (int): Output dimensions.
    '''
    def __init__(self, input_dim: int=512, hidden_layers: list=[512,512,128], output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.nn_layers = nn.ModuleList()
        
        self.nn_layers.append(lmn.LipschitzLinear(input_dim, hidden_layers[0], kind="one-inf"))
        self.nn_layers.append(lmn.GroupSort(2))
        
        for i in range(1,len(hidden_layers)):
            self.nn_layers.append(lmn.LipschitzLinear(hidden_layers[i-1], hidden_layers[i], kind="inf"))
            self.nn_layers.append(lmn.GroupSort(2))


        self.nn_layers.append(lmn.LipschitzLinear(hidden_layers[-1], output_dim, kind="inf"))
        
    def forward(self, x):
        """
        Forward pass of the MLP module.
        
        Args:
            - x (Pytorch Tensor): Input tensor.
        
        Returns:
            - Output tensor after passing through the MLP layers.
            
        """
        for layer in self.nn_layers:
            x = layer(x)
            
        return x  

class MonotonicMLP(nn.Module):
    '''
    Combines LipschitzMLP with monotonic wrapper and adds sigmoig at the end.

    Attributes:
        lipschitz_mlp (nn.Module): Lipschitz MLP.
        monotonic_constraints (list): Specifies which variables will be monotonic.
        lipschitz_const (float): Lipschitz constant that determines the maximum value of the derivative.
    '''
    def __init__(self, lipschitz_mlp, monotonic_constraints, lipschitz_const):
        super().__init__()
        self.mono = lmn.MonotonicWrapper(
            lipschitz_mlp,
            monotonic_constraints=monotonic_constraints,
            lipschitz_const=lipschitz_const
        )
        
    def forward(self, x):
        return torch.sigmoid(0.3 * self.mono(x))    
    
class ConvBlock(nn.Module):
    """
    A custom convolutional block with Pytorch that consists of two convolution layers.
    
    Attributes:
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
        
        Attributes:
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
    
    Attributes:
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
    
    Attributes:
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

class SimpleViT(nn.Module):
    """
    A simple vision Transformer that patchifies images, considers patches as 1D array, and encodes them
    with a linear layer.

    Arguments
    ---------
        - in_channels (int): number of input channels.
        - d_model (int): dimensionality of transformer latent space.
        - image_size (int): pixel size of input images.
        - patch_size (int): pixel size of patches.
        - n_head (int): number of heads in the attention layer.
        - dim_feedfowrard (int): hidden dimensionality of MLP in attention layer.
        - num_layers (int): number of attention layers in series.
    """
    def __init__(
        self,
        in_channels=6,
        d_model=256,
        image_size=36,
        patch_size=4,
        nhead=8,
        dim_feedforward=1024,
        num_layers=6,
    ):
        super().__init__()
        # 2D convolution does patching + embedding in one go.
        self.embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        n_patches = (image_size // patch_size) ** 2

        # cls_token is what's uses as embedding vector.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches+1, d_model))

        # this layer contains LayerNorms
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # token sequences of shape (N_batch, d_model, sqrt(n_patches), sqrt(n_patches))
        x = self.embed(x)

        # Flattened patches of shape (N_batch, n_patches, d_model)
        x = x.flatten(2).transpose(1,2)

        # N_batch copies of the cls_token.
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        # concat to the other tokens to get (N_batch, n_patches+1, d_model)
        x = torch.cat((x, cls), dim=1)

        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x)

        return x[:,0]