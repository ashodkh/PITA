import numpy as np
import torch

class JitterCrop:
    def __init__(self, output_dim, jitter_lim=None):
        self.output_dim = output_dim
        self.jitter_lim = jitter_lim
        self.half_dim = output_dim//2
        
    def __call__(self, image):
        # assumed that image shape is N,C,pixels,pixels
        center_x = image.shape[2]//2
        center_y = image.shape[2]//2
        if self.jitter_lim:
            center_x = center_x + int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
            center_y = center_y + int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
        
        return image[:,:, center_y-self.half_dim:center_y+self.half_dim,
                     center_x-self.half_dim:center_x+self.half_dim]
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.shape, device=tensor.device, dtype=tensor.dtype)\
               * torch.tensor(self.std[None,:,None,None],device=tensor.device, dtype=tensor.dtype) + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)