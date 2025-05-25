import h5py
import torch
import random
import argparse
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument("--h5_file", type=str, required=True, default='data/test_scene_qcs100.h5')
args = parser.parse_args()

pcd_occ_h5 = h5py.File(args.h5_file, "r")
quad_chunk_size = int(args.h5_file.split('.')[0].split('qcs')[-1])
rand_idx = random.randint(0, pcd_occ_h5['near_surface_occ'].shape[0])
pcd = pcd_occ_h5['pcd'][rand_idx]
num_actual_pcd = pcd_occ_h5['num_actual_pcd'][rand_idx]

num_actual_pos = pcd_occ_h5['num_actual_pos'][rand_idx]
pos_surface_pos = pcd_occ_h5['pos_surface_pos'][rand_idx]

num_actual_neg = pcd_occ_h5['num_actual_neg'][rand_idx]
neg_surface_pos = pcd_occ_h5['neg_surface_pos'][rand_idx]

num_actual_near = pcd_occ_h5['num_actual_near'][rand_idx]
near_surface_pos = pcd_occ_h5['near_surface_pos'][rand_idx]
near_surface_occ = pcd_occ_h5['near_surface_occ'][rand_idx]

counter = 0
quad_pcd = []
quad_pos_pos = []
quad_neg_pos = []
quad_near_pos = []
quad_near_occ = []
for i in range(int(np.sqrt(4))):
    for j in range(int(np.sqrt(4))):
        pcd[counter, :num_actual_pcd[counter], 0] = pcd[counter, :num_actual_pcd[counter], 0] + i * 2
        pcd[counter, :num_actual_pcd[counter], 2] = pcd[counter, :num_actual_pcd[counter], 2] + j * 2
        quad_pcd.append(pcd[counter, :num_actual_pcd[counter]])

        pos_surface_pos[counter, :num_actual_pos[counter], 0] = pos_surface_pos[counter, :num_actual_pos[counter], 0] + i * 2
        pos_surface_pos[counter, :num_actual_pos[counter], 2] = pos_surface_pos[counter, :num_actual_pos[counter], 2] + j * 2
        quad_pos_pos.append(pos_surface_pos[counter, :num_actual_pos[counter]])

        neg_surface_pos[counter, :num_actual_neg[counter], 0] = neg_surface_pos[counter, :num_actual_neg[counter], 0] + i * 2
        neg_surface_pos[counter, :num_actual_neg[counter], 2] = neg_surface_pos[counter, :num_actual_neg[counter], 2] + j * 2
        quad_neg_pos.append(neg_surface_pos[counter, :num_actual_neg[counter]])

        near_surface_pos[counter, :num_actual_near[counter], 0] = near_surface_pos[counter, :num_actual_near[counter], 0] + i * 2
        near_surface_pos[counter, :num_actual_near[counter], 2] = near_surface_pos[counter, :num_actual_near[counter], 2] + j * 2
        quad_near_pos.append(near_surface_pos[counter, :num_actual_near[counter]])
        quad_near_occ.append(near_surface_occ[counter, :num_actual_near[counter]])

        counter += 1

print('Visualizing quad chunk sampled point cloud:')
quad_pcd = np.concatenate(quad_pcd)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(quad_pcd)
o3d.visualization.draw_geometries([pcd])

print('Visualizing quad chunk sampled volumetric empty (green) and occupied (red) occupancies:')
quad_pos_pos = np.concatenate(quad_pos_pos)
quad_neg_pos = np.concatenate(quad_neg_pos)
pos_pcd = o3d.geometry.PointCloud()
pos_pcd.points = o3d.utility.Vector3dVector(quad_pos_pos)
pos_pcd.paint_uniform_color([1, 0, 0])
neg_pcd = o3d.geometry.PointCloud()
neg_pcd.points = o3d.utility.Vector3dVector(quad_neg_pos)
neg_pcd.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([pos_pcd, neg_pcd])

print('Visualizing quad chunk sampled surface empty (green) and occupied (red) occupancies:')
quad_near_pos = torch.from_numpy(np.concatenate(quad_near_pos))
quad_near_occ = torch.from_numpy(np.concatenate(quad_near_occ))
near_pos_select = torch.stack(torch.where(quad_near_occ == 1)).T
near_pos_pos = quad_near_pos[near_pos_select.squeeze()]
near_neg_select = torch.stack(torch.where(quad_near_occ == 0)).T
near_neg_pos = quad_near_pos[near_neg_select.squeeze()]
pos_pcd = o3d.geometry.PointCloud()
pos_pcd.points = o3d.utility.Vector3dVector(near_pos_pos.numpy())
pos_pcd.paint_uniform_color([1, 0, 0])
neg_pcd = o3d.geometry.PointCloud()
neg_pcd.points = o3d.utility.Vector3dVector(near_neg_pos.numpy())
neg_pcd.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([pos_pcd, neg_pcd])
