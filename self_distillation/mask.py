import torch
from torch import Tensor
from self_distillation.print_color import *
from torch_geometric.data import HeteroData


def mask_scenario(input: HeteroData):
    masked_input = input.clone()
    agent_mask_shape = input['agent']['valid_mask'].shape
    agent_mask = generate_random_binary_mask(agent_mask_shape, 0.5)

    map_mask_shape = input['map_polygon']['orientation'].shape
    map_mask =  generate_random_binary_mask(map_mask_shape, 0.5)

    #* Mask agents
    masked_input['agent']['position'][agent_mask] = 0.0
    masked_input['agent']['heading'][agent_mask] = 0.0
    masked_input['agent']['velocity'][agent_mask] = 0.0
    masked_input['agent']['valid_mask'][agent_mask] = False

    #* Mask map
    # TODO masked_input['map_polygon']

    return masked_input



def generate_random_binary_mask(shape, true_ratio):
    """
    Generate a random binary mask of the given shape with the specified ratio of `True` values (1s).

    Args:
        shape (tuple): Shape of the desired mask (e.g., (2, 3) for a 2x3 mask).
        true_ratio (float): Ratio of `True/ masked values (1s) in the mask. Must be between 0 and 1.

    Returns:
        torch.Tensor: Binary mask with the specified shape and ratio of `True` values.
    """
    assert 0 <= true_ratio <= 1, "true_ratio must be between 0 and 1"
    
    num_elements = torch.prod(torch.tensor(shape))  # Total number of elements in the mask
    num_true = int(num_elements * true_ratio)   # Number of `True` values (1s) to include
    flat_mask = torch.cat([torch.ones(num_true), torch.zeros(num_elements - num_true)])
    permuted_mask = flat_mask[torch.randperm(num_elements)]     # Shuffle the tensor to randomly distribute the `True` values    
    binary_mask = permuted_mask.view(*shape).bool()
    
    return binary_mask
