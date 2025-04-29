import os

from matplotlib import pyplot as plt
import numpy as np
import torch

use_cuda = torch.cuda.is_available()
print('Using cuda:{}'.format(use_cuda))
torch_cuda = torch.cuda if use_cuda else torch
device = torch.device('cuda' if use_cuda else 'cpu')

def inspect_loader_coordinates(loader):
    min_coords = torch.tensor([float('inf')] * 3)  
    max_coords = torch.tensor([-float('inf')] * 3)
    
    for batch in loader:
        points = batch[0]  
        batch_min, _ = torch.min(points.view(-1, 3), dim=0) 
        batch_max, _ = torch.max(points.view(-1, 3), dim=0)  
        
        min_coords = torch.minimum(min_coords, batch_min)
        max_coords = torch.maximum(max_coords, batch_max)
    
    print(f"coordinate range:\n  X: [{min_coords[0]:.3f}, {max_coords[0]:.3f}]\n"
          f"  Y: [{min_coords[1]:.3f}, {max_coords[1]:.3f}]\n"
          f"  Z: [{min_coords[2]:.3f}, {max_coords[2]:.3f}]")

def save_coord_histograms(loader, save_dir="distribution_plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    all_points = np.concatenate([batch[0].numpy().reshape(-1, 3) for batch in loader], axis=0)
    
    plt.figure(figsize=(15, 5))
    for i, axis in enumerate(['X', 'Y', 'Z']):
        plt.subplot(1, 3, i+1)
        plt.hist(all_points[:, i], bins=100, alpha=0.7, color=['r','g','b'][i])
        plt.title(f'{axis} Axis Distribution')
        plt.xlabel('Normalized Value')
        plt.ylabel('Count')
    
    plot_path = os.path.join(save_dir, "coord_histograms.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram to {plot_path}")


def advanced_coordinate_analysis(loader):
    points = torch.cat([batch[0].view(-1, 3) for batch in loader], dim=0)
    
    stats = {
        'mean': torch.mean(points, dim=0),
        'std': torch.std(points, dim=0),
        'median': torch.median(points, dim=0).values,
        'mad': torch.median(torch.abs(points - torch.median(points, dim=0).values), dim=0).values  # absolute deviation
    }
    
    print("statistic:")
    for axis in ['X', 'Y', 'Z']:
        idx = ['X', 'Y', 'Z'].index(axis)
        print(f"{axis}axis: mean={stats['mean'][idx]:.3f}, "
              f"standard={stats['std'][idx]:.3f}, "
              f"median={stats['median'][idx]:.3f}, "
              f"MAD={stats['mad'][idx]:.3f}")
