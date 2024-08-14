import torch
from torch import Tensor
from self_distillation.print_color import *
from torch_geometric.data import HeteroData

def flip(input: HeteroData):
    """ Flip around the x-axis """
    augmented_input = input.clone()

    #* Flip agnet's y-coodinates
    augmented_input['agent']['position'][..., 1] *= -1
    augmented_input['agent']['heading'] *= -1
    augmented_input['agent']['velocity'][..., 1] *= -1
    augmented_input['agent']['target'][..., 1] *= -1    # y
    augmented_input['agent']['target'][..., 3] *= -1    # theta

    #* Flip the map
    augmented_input['map_polygon']['position'][..., 1] *= -1
    augmented_input['map_polygon']['orientation'] *= -1
    augmented_input['map_point']['position'][..., 1] *= -1
    augmented_input['map_point']['orientation'] *= -1

    return augmented_input
