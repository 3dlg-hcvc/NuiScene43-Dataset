import os
import time
import glob
import json
import torch
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.signal import convolve2d

parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument("--num_splits", type=int, default=4)
parser.add_argument("--quad_chunk_size", type=int, default=100)
parser.add_argument("--split_idx", type=int, required=True)
parser.add_argument("--num_occ_sample", type=int, default=75000)
parser.add_argument("--sample_json", type=str, default='metadata/test_scene.json')
args = parser.parse_args()

with open(args.sample_json, "r") as f:
    sample_scenes = json.load(f)

output_dir = 'data/{}_qcs{}'.format(args.sample_json.split('/')[-1].split('.')[0], args.quad_chunk_size)
os.makedirs(output_dir, exist_ok=True)

quad_chunk_size = args.quad_chunk_size
samp_mask_quad_chunk_size = 80
num_neg_sample = args.num_occ_sample
num_pos_sample = args.num_occ_sample
num_near_sample = args.num_occ_sample

filter_model_ids = list(sample_scenes.keys())
mask_files = sorted(glob.glob('data/nuiscene43/*/*sample_masks.npy'))
filtered_mask_files = []
for mask_file in mask_files:
    for filter_model_id in filter_model_ids:
        if filter_model_id in mask_file:
            filtered_mask_files.append(mask_file)
            break
mask_files = filtered_mask_files
mask_files = [mask_file for mask_file in mask_files if 'special' not in mask_file]
mask_files = sorted(mask_files)

occ_files = glob.glob('data/nuiscene43/*/*occ_res*.npz')

valid_model_ids = [path.split('/')[1].split('.')[0] for path in occ_files]
mask_files = [mask_file for mask_file in mask_files if mask_file.split('/')[1] in valid_model_ids]

model_occ_file = {}
model_grid_resolutions = {}
for occ_file in occ_files:
    model_occ_file[occ_file.split('/')[-1].split('.')[0]] = occ_file
    model_grid_resolutions[occ_file.split('/')[-1].split('.')[0]] = int(occ_file.split('/')[-1].split('.')[1].split('_')[-1][3:])

def sample_chunk(_mask_files):
    for mask_file in tqdm(_mask_files):
        new_num_sample = sample_scenes[mask_file.split('/')[-1].split('_sample_masks')[0]]

        # Load sampling masks
        masks = np.load(mask_file, allow_pickle=True).item()
        model_id = mask_file.split('/')[-1].split('_sample_masks')[0]
        os.makedirs(os.path.join(output_dir, model_id), exist_ok=True)
        sample_mask = masks['sample_mask']
        depth_var_map = masks['depth_var_map']

        # Kernel again to reduce sample mask size to large chunks size
        kernel = np.ones((quad_chunk_size - samp_mask_quad_chunk_size, quad_chunk_size - samp_mask_quad_chunk_size), dtype=int)
        convolved_map = convolve2d(sample_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        sample_mask = convolved_map / kernel.sum()
        sample_mask = (sample_mask >= 1)

        # Basic depth threshold
        depth_var_threshold = 2.5
        depth_var_mask = depth_var_map > depth_var_threshold
        sample_mask = np.logical_and(sample_mask, depth_var_mask)

        # Load pcd, occupancy and surface occupancy
        occ = np.load(model_occ_file[model_id])['occ']
        pcd = np.load(glob.glob('data/nuiscene43/{}/{}.pcd_*.npz'.format(model_id, model_id))[0])['pcd']

        pcd = torch.from_numpy(pcd)
        pcd[:, 2] = occ.shape[2] - pcd[:, 2]

        # Get the valid sample coordinates and sample new_num_sample points
        valid_sample_coordinates = np.argwhere(sample_mask == 1)

        start = time.time()
        print('Processing FPS of {}...'.format(model_id), 'Num valid samples:', valid_sample_coordinates.shape[0])
        if valid_sample_coordinates.shape[0] < int(new_num_sample * 1.5):
            sampled_coordinates = valid_sample_coordinates
        else:        
            fake_pcd = np.hstack([valid_sample_coordinates, np.ones((valid_sample_coordinates.shape[0], 1))])
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(fake_pcd)
            sampled_coordinates = point_cloud.farthest_point_down_sample(int(new_num_sample * 1.5))
            sampled_coordinates = np.asarray(sampled_coordinates.points)[:, :2]
            sampled_coordinates = sampled_coordinates.astype(np.int64)
            np.random.shuffle(sampled_coordinates)
        print('Done FPS of {}...'.format(model_id), 'Time:', time.time() - start)

        counter = 0
        for i in tqdm(range(sampled_coordinates.shape[0])):
            sample_x = sampled_coordinates[i, 0]
            sample_y = sampled_coordinates[i, 1]

            crop_occ = np.flip(occ, axis=(2))[sample_x-quad_chunk_size//2:sample_x+quad_chunk_size//2, :, sample_y-quad_chunk_size//2:sample_y+quad_chunk_size//2]
            crop_occ = np.flip(crop_occ, axis=(2))

            crop_surf_occ = np.flip(occ, axis=(2))[sample_x-quad_chunk_size//2:sample_x+quad_chunk_size//2, :, sample_y-quad_chunk_size//2:sample_y+quad_chunk_size//2]
            crop_surf_occ = np.flip(crop_surf_occ, axis=(2))
            crop_surf_occ = torch.from_numpy(crop_surf_occ)

            crop_pcd = pcd[(pcd[:, 0] >= sample_x-quad_chunk_size//2) & (pcd[:, 0] <= sample_x+quad_chunk_size//2) &
                        (pcd[:, 2] >= sample_y-quad_chunk_size//2) & (pcd[:, 2] <= sample_y+quad_chunk_size//2)]
            if crop_pcd.shape[0] < 10000:
                continue
            crop_pcd[:, 0] = crop_pcd[:, 0] - sample_x + quad_chunk_size//2
            crop_pcd[:, 2] = crop_pcd[:, 2] - sample_y + quad_chunk_size//2
            crop_pcd[:, 2] = crop_surf_occ.shape[2] - crop_pcd[:, 2]

            max_height = int(crop_pcd.max(0)[0][1].item() + quad_chunk_size // 4)
            crop_occ = crop_occ[:, :max_height]
            crop_surf_occ = crop_surf_occ[:, :max_height]

            # Add jitter to the point cloud as use as near surface samples
            jitter = np.random.uniform(low=-2, high=2, size=crop_pcd.shape)
            crop_pcd_jitter = (crop_pcd + jitter).long()
            crop_pcd_jitter[:, 0] = np.clip(crop_pcd_jitter[:, 0], 0, crop_surf_occ.shape[0] - 1)
            crop_pcd_jitter[:, 1] = np.clip(crop_pcd_jitter[:, 1], 0, crop_surf_occ.shape[1] - 1)
            crop_pcd_jitter[:, 2] = np.clip(crop_pcd_jitter[:, 2], 0, crop_surf_occ.shape[2] - 1)

            # Get the new surface samples
            near_surface_indices = np.random.choice(crop_pcd_jitter.shape[0], num_near_sample, replace=crop_pcd_jitter.shape[0] < num_near_sample)
            near_surface_pos = crop_pcd_jitter[near_surface_indices].numpy().astype(np.uint16)
            x, y, z = near_surface_pos[:, 0], near_surface_pos[:, 1], near_surface_pos[:, 2]
            near_surface_occ = crop_occ[x, y, z]
            num_actual_near = min(crop_pcd_jitter.shape[0], num_near_sample)

            pos_surface_indices = torch.stack(torch.where(crop_surf_occ == 1)).T
            if pos_surface_indices.shape[0] < num_pos_sample:
                short_samples = np.arange(pos_surface_indices.shape[0])
                expand_samples = np.random.choice(short_samples, size=num_pos_sample - pos_surface_indices.shape[0], replace=True)
                pos_surface_indices_samples = np.concatenate([short_samples, expand_samples])
            else:
                pos_surface_indices_samples = np.random.choice(pos_surface_indices.shape[0], size=num_pos_sample, replace=False)

            num_actual_pos = min(pos_surface_indices.shape[0], num_pos_sample)
            pos_surface_pos = pos_surface_indices[pos_surface_indices_samples].numpy().astype(np.uint16)
            pos_surface_occ = crop_occ[pos_surface_pos[:, 0], pos_surface_pos[:, 1], pos_surface_pos[:, 2]]

            neg_surface_indices = torch.stack(torch.where(crop_surf_occ == 0)).T
            if neg_surface_indices.shape[0] < num_neg_sample:
                short_samples = np.arange(neg_surface_indices.shape[0])
                expand_samples = np.random.choice(short_samples, size=num_neg_sample - neg_surface_indices.shape[0], replace=True)
                neg_surface_indices_samples = np.concatenate([short_samples, expand_samples])
            else:
                neg_surface_indices_samples = np.random.choice(neg_surface_indices.shape[0], size=num_neg_sample, replace=False)
            
            num_actual_neg = min(neg_surface_indices.shape[0], num_neg_sample)
            neg_surface_pos = neg_surface_indices[neg_surface_indices_samples].numpy().astype(np.uint16)
            neg_surface_occ = crop_occ[neg_surface_pos[:, 0], neg_surface_pos[:, 1], neg_surface_pos[:, 2]]

            output = {}

            output['pcd'] = crop_pcd.numpy().astype(np.uint16)
            output['sample_x'] = sample_x
            output['sample_y'] = sample_y

            output['num_actual_near'] = num_actual_near
            output['near_surface_pos'] = near_surface_pos
            output['near_surface_occ'] = near_surface_occ.astype(np.uint8)

            output['num_actual_pos'] = num_actual_pos
            output['pos_surface_pos'] = pos_surface_pos
            assert pos_surface_occ.sum() == pos_surface_occ.shape[0]

            output['num_actual_neg'] = num_actual_neg
            output['neg_surface_pos'] = neg_surface_pos
            assert neg_surface_occ.sum() == 0

            np.savez_compressed(os.path.join(output_dir, model_id, 'chunk_{}.npz'.format(counter)), **output)

            counter += 1
            if counter == new_num_sample:
                break

split_idx = args.split_idx
num_splits = args.num_splits
if num_splits > len(mask_files):
    num_splits = len(mask_files)
chunk_per_split = len(mask_files) // num_splits

if split_idx == num_splits - 1:
    sample_chunk(mask_files[split_idx * chunk_per_split:])
else:
    sample_chunk(mask_files[split_idx * chunk_per_split:(split_idx + 1) * chunk_per_split])
