import numpy as np
import torch
    
class ReddeningTransform:
    def __init__(self, R=None, redden_aug=False, ebv_max=0.5):
        self.R = np.array(R)
        self.redden_aug = redden_aug 
        self.ebv_max = ebv_max

    def __call__(self, data):
        image, ebv = data
        # assumed that image shape is C,pixels,pixels
        assert self.R.shape == image.shape[0]
        # Detect input type
        is_torch = isinstance(image, torch.Tensor)
        if is_torch:
            R = torch.as_tensor(self.R, dtype=image.dtype, device=image.device)
        else:
            R = self.R.astype(image.dtype)
            
        if not is_torch:  
            ebv = ebv.astype(image.dtype)
            
        if ebv is not None:
            # Deredden image
            true_ext = R * ebv
            factor = 10.0 ** (true_ext[:, None, None] / 2.5)
            image = image * factor
        elif self.redden_aug:
            new_ebv = np.random.uniform(0, self.ebv_max)
            factor = 10.0 ** (-(R * new_ebv)[:, None, None] / 2.5)
            if is_torch:
                factor = torch.as_tensor(factor, dtype=image.dtype, device=image.device)
            image = image * factor

        return image
