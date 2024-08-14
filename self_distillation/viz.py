
import matplotlib.pyplot as plt

from av2.map.map_api import ArgoverseStaticMap, LaneSegment
from typing import Final, List, Optional, Sequence, Set, Tuple
from av2.utils.typing import NDArrayFloat, NDArrayInt

import av2.geometry.interpolate as interp_utils
from time import sleep
import torch
from matplotlib.collections import LineCollection
import numpy as np

from self_distillation.print_color import * 

def plot_scenario(data, name = "test"):
    _plot_map(data)
    _plot_traj(data)
    
    plt.savefig(f'fig/{name}.png', format="png")
    plt.close()
    pc('fig_saved')

def _plot_map(data):
    points = data['map_point']['position']

    for p in points:
        x = p[0]
        y = p[1]
        plt.scatter(x.cpu().detach().numpy(), y.cpu().detach().numpy(), c='gray', s=0.7, alpha=0.3)

def _plot_traj(data):
    padding_masks = data['agent']['valid_mask']
    positions = data['agent']['position']
    # batches = data['agent']['batch']

    for agent_idx in range(positions.shape[0]):
        # if batches[agent_idx] != 0:
        #     break
        past_padding_mask = padding_masks[agent_idx, :50]
        future_padding_mask = padding_masks[agent_idx, 50:]

        x_past = positions[agent_idx, :50, 0][past_padding_mask]
        x_future = positions[agent_idx, 50:, 0][future_padding_mask]

        y_past = positions[agent_idx, :50, 1][past_padding_mask]
        y_future = positions[agent_idx, 50:, 1][future_padding_mask]
        
        plt.scatter(x_past.cpu().detach().numpy(), y_past.cpu().detach().numpy(), c='b',  s=0.7)
        plt.scatter(x_future.cpu().detach().numpy(), y_future.cpu().detach().numpy(), c='r', s= 0.7)