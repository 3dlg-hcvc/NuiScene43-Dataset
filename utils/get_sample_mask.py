import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import generic_filter

parser = argparse.ArgumentParser()
parser.add_argument("--quad_chunk_size", type=int, default=80)
args = parser.parse_args()

margin = 2
sdf_file = '3b61335c2a004a9ea31c8dab59471222.sdf_res1162.npz'
chunk_size = (args.quad_chunk_size + margin, args.quad_chunk_size + margin)

model_id = sdf_file.split('/')[-1].split('.')[0]
output_dir = os.path.join('data/scene_mask', model_id)
os.makedirs(output_dir, exist_ok=True)

sdf_grid = np.load(sdf_file)['sdf']
occ_grid = (sdf_grid <= 0).astype(np.int32)

depth_map = np.full((occ_grid.shape[0], occ_grid.shape[-1]), np.max(occ_grid.shape)+1, dtype=int)
first_one_indices = np.argmax(np.flip(occ_grid, axis=(1, 2)), axis=1)
mask = np.any(np.flip(occ_grid, axis=(1, 2)), axis=1)
depth_map[mask] = first_one_indices[mask]
alpha_mask = (depth_map < np.max(occ_grid.shape)).astype(np.int32)

kernel = np.ones(chunk_size, dtype=int)
convolved_map = convolve2d(alpha_mask, kernel, mode='same', boundary='fill', fillvalue=0)
sample_mask = convolved_map / kernel.sum()
sample_mask = (sample_mask >= 1)

plt.figure(figsize=(10, 10))
plt.imshow(depth_map, cmap='viridis')
plt.colorbar(label='Depth (y-axis index)')
plt.title('Depth Map Projection')
plt.axis('off')
plt.savefig(os.path.join(output_dir, 'depth_map.png'), dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 10))
plt.imshow(alpha_mask, cmap='gray', interpolation='nearest')
plt.title('Binary Occupancy Map')
plt.colorbar(label='Occupied (1) / Empty (0)')
plt.axis('off')
plt.savefig(os.path.join(output_dir, 'alpha_map.png'), dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 10))
plt.imshow(sample_mask, cmap='gray', interpolation='nearest')
plt.title('Binary Occupancy Map')
plt.colorbar(label='Occupied (1) / Empty (0)')
plt.axis('off')
plt.savefig(os.path.join(output_dir, 'sample_mask.png'), dpi=300, bbox_inches='tight')

def mean_absolute_deviation(window):
    mean_val = np.mean(window)
    return np.mean(np.abs(window - mean_val))

depth_variation_map = generic_filter(depth_map, mean_absolute_deviation, size=chunk_size)
depth_variation_map *= sample_mask

plt.figure(figsize=(10, 10))
plt.imshow(depth_variation_map, cmap='inferno', interpolation='nearest')
plt.title('Depth Variation (Max - Min) in 41x41 Window')
plt.colorbar(label='Depth Variation')
plt.axis('off')
plt.savefig(os.path.join(output_dir, 'depth_var_map.png'), dpi=300, bbox_inches='tight')

output = {}
output['depth_map'] = depth_map
output['sample_mask'] = sample_mask
output['depth_var_map'] = depth_variation_map
np.save(os.path.join(output_dir, 'sample_masks.npy'), output)
