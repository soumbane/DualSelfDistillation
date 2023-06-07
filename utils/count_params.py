# Count the number of parameters of the model
import torch

def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += torch.numel(param)

    return total_params
    
