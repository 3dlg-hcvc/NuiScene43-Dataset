import glob
import trimesh
import argparse
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--scene_id', type=str, default='3b61335c2a004a9ea31c8dab59471222')
args = parser.parse_args()

occ_file = glob.glob('./data/nuiscene43/{}/*.occ*'.format(args.scene_id))[0]
occ_grid = np.load(occ_file)['occ']

mask_file = glob.glob('./data/nuiscene43/{}/*sample_masks*'.format(args.scene_id))[0]
masks = np.load(mask_file, allow_pickle=True).item()
sample_mask = masks['sample_mask']
depth_var_map = masks['depth_var_map']

vertices, faces, _, _ = skimage.measure.marching_cubes(occ_grid, 0)
mesh = trimesh.Trimesh(vertices, faces)
mesh.export('{}.obj'.format(args.scene_id))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(sample_mask, cmap='gray')
axs[0].set_title('Mask (True/False)')
axs[0].axis('off')
im = axs[1].imshow(depth_var_map, cmap='viridis')
axs[1].set_title('Depth Variance')
axs[1].axis('off')
cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
cbar.set_label('Variance')
plt.tight_layout()
plt.savefig("{}_maps.png".format(args.scene_id), dpi=300, bbox_inches='tight')
plt.close()
