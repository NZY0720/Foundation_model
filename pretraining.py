import torch

class MaskedETSModeling:
    @staticmethod
    def mask_data(data, mask_ratio=0.4, causal=False):
        masked_data = data.clone()
        if causal:
            for i in range(data.size(1)):
                if torch.rand(1).item() < mask_ratio:
                    masked_data[:, i:] = 0
        else:
            mask = torch.rand(data.shape) < mask_ratio
            masked_data[mask] = 0
        return masked_data
