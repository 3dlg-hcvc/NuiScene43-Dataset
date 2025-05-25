import glob
import json
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--quad_chunk_size", type=int, default=100)
parser.add_argument("--num_pcd_sample_per_chunk", type=int, default=15000)
parser.add_argument("--num_occ_sample_per_chunk", type=int, default=12500)
parser.add_argument("--sample_json", type=str, default='metadata/test_scene.json')
args = parser.parse_args()

class ChunksDataset(Dataset):
    def __init__(self, raw_chunks_paths, quad_chunk_size):
        self.raw_chunks_paths = raw_chunks_paths

        self.num_chunks = 4
        self.quad_chunk_size = quad_chunk_size

        self.num_sample_pcd = args.num_pcd_sample_per_chunk
        self.num_sample_pos = args.num_occ_sample_per_chunk
        self.num_sample_neg = args.num_occ_sample_per_chunk
        self.num_sample_near = args.num_occ_sample_per_chunk

    def __len__(self):
        return len(self.raw_chunks_paths)

    def __getitem__(self, idx):
        chunk_path = self.raw_chunks_paths[idx]
        chunk = np.load(chunk_path)

        pcd = chunk['pcd']
        pcd = pcd[(pcd[:, 0] < self.quad_chunk_size) & (pcd[:, 2] < self.quad_chunk_size)]

        num_actual_pos = chunk['num_actual_pos']
        pos_surface_pos = chunk['pos_surface_pos']
        pos_surface_pos = pos_surface_pos[:num_actual_pos]

        num_actual_neg = chunk['num_actual_neg']
        neg_surface_pos = chunk['neg_surface_pos']
        neg_surface_pos = neg_surface_pos[:num_actual_neg]

        num_actual_near = chunk['num_actual_near']
        near_surface_pos = chunk['near_surface_pos']
        near_surface_occ = chunk['near_surface_occ']
        near_surface_pos = near_surface_pos[:num_actual_near]
        near_surface_occ = near_surface_occ[:num_actual_near]

        # Convert data into torch tensors
        pcd = torch.from_numpy(pcd.astype(np.float32))
        pos_surface_pos = torch.from_numpy(pos_surface_pos.astype(np.float32))
        neg_surface_pos = torch.from_numpy(neg_surface_pos.astype(np.float32))
        near_surface_pos = torch.from_numpy(near_surface_pos.astype(np.float32))
        near_surface_occ = torch.from_numpy(near_surface_occ)

        large_chunk_pcds = []
        large_chunk_actual_num_pcds = []

        large_chunk_pos_occ_queries = []
        large_chunk_actual_num_pos = []

        large_chunk_neg_occ_queries = []
        large_chunk_actual_num_neg = []

        large_chunk_near_occs = []
        large_chunk_near_occ_queries = []
        large_chunk_actual_num_near = []

        offset = 1 / np.sqrt(self.num_chunks)
        for i in range(int(np.sqrt(self.num_chunks))):
            for j in range(int(np.sqrt(self.num_chunks))):
                x = i * offset
                z = j * offset

                x_range = np.array([x, x + offset]) * self.quad_chunk_size
                z_range = np.array([z, z + offset]) * self.quad_chunk_size

                filtered_pcd = pcd[
                    (pcd[:, 0] >= x_range[0]) & (pcd[:, 0] <= x_range[1]) &
                    (pcd[:, 2] >= z_range[0]) & (pcd[:, 2] <= z_range[1])
                ]
                filtered_pcd[:, 0] = (filtered_pcd[:, 0] - x_range[0]) / (self.quad_chunk_size / np.sqrt(self.num_chunks))
                filtered_pcd[:, 1] = filtered_pcd[:, 1] / (self.quad_chunk_size / np.sqrt(self.num_chunks))
                filtered_pcd[:, 2] = (filtered_pcd[:, 2] - z_range[0]) / (self.quad_chunk_size / np.sqrt(self.num_chunks))
                filtered_pcd = filtered_pcd * 2 - 1

                large_chunk_actual_num_pcds.append(min(self.num_sample_pcd, filtered_pcd.shape[0]))
                ind = np.random.choice(filtered_pcd.shape[0], self.num_sample_pcd, replace=(self.num_sample_pcd > filtered_pcd.shape[0]))
                filtered_pcd = filtered_pcd[ind]
                large_chunk_pcds.append(filtered_pcd)

                pos_mask = (pos_surface_pos[:, 0] >= x_range[0]) & (pos_surface_pos[:, 0] <= x_range[1]) & \
                            (pos_surface_pos[:, 2] >= z_range[0]) & (pos_surface_pos[:, 2] <= z_range[1])
                filtered_pos = pos_surface_pos[pos_mask]
                filtered_pos[:, 0] = (filtered_pos[:, 0] - x_range[0]) / (self.quad_chunk_size / np.sqrt(self.num_chunks))
                filtered_pos[:, 1] = filtered_pos[:, 1] / (self.quad_chunk_size / np.sqrt(self.num_chunks))
                filtered_pos[:, 2] = (filtered_pos[:, 2] - z_range[0]) / (self.quad_chunk_size / np.sqrt(self.num_chunks))
                filtered_pos = filtered_pos * 2 - 1

                large_chunk_actual_num_pos.append(min(self.num_sample_pos, filtered_pos.shape[0]))
                if filtered_pos.shape[0] < self.num_sample_pos:
                    short_samples = np.arange(filtered_pos.shape[0])
                    expand_samples = np.random.choice(short_samples, size=self.num_sample_pos - filtered_pos.shape[0], replace=True)
                    pos_surface_indices_samples = np.concatenate([short_samples, expand_samples])
                else:
                    pos_surface_indices_samples = np.random.choice(filtered_pos.shape[0], size=self.num_sample_pos, replace=False)
                filtered_pos = filtered_pos[pos_surface_indices_samples]
                large_chunk_pos_occ_queries.append(filtered_pos)

                neg_mask = (neg_surface_pos[:, 0] >= x_range[0]) & (neg_surface_pos[:, 0] <= x_range[1]) &\
                           (neg_surface_pos[:, 2] >= z_range[0]) & (neg_surface_pos[:, 2] <= z_range[1])
                filtered_neg = neg_surface_pos[neg_mask]
                filtered_neg[:, 0] = (filtered_neg[:, 0] - x_range[0]) / (self.quad_chunk_size / np.sqrt(self.num_chunks))
                filtered_neg[:, 1] = filtered_neg[:, 1] / (self.quad_chunk_size / np.sqrt(self.num_chunks))
                filtered_neg[:, 2] = (filtered_neg[:, 2] - z_range[0]) / (self.quad_chunk_size / np.sqrt(self.num_chunks))
                filtered_neg = filtered_neg * 2 - 1

                large_chunk_actual_num_neg.append(min(self.num_sample_neg, filtered_neg.shape[0]))
                if filtered_neg.shape[0] < self.num_sample_neg:
                    short_samples = np.arange(filtered_neg.shape[0])
                    expand_samples = np.random.choice(short_samples, size=self.num_sample_neg - filtered_neg.shape[0], replace=True)
                    neg_surface_indices_samples = np.concatenate([short_samples, expand_samples])
                else:
                    neg_surface_indices_samples = np.random.choice(filtered_neg.shape[0], size=self.num_sample_neg, replace=False)
                filtered_neg = filtered_neg[neg_surface_indices_samples]
                large_chunk_neg_occ_queries.append(filtered_neg)

                near_mask = (near_surface_pos[:, 0] >= x_range[0]) & (near_surface_pos[:, 0] <= x_range[1]) & \
                           (near_surface_pos[:, 2] >= z_range[0]) & (near_surface_pos[:, 2] <= z_range[1])
                filtered_near = near_surface_pos[near_mask]
                filtered_near_occ = near_surface_occ[near_mask]
                filtered_near[:, 0] = (filtered_near[:, 0] - x_range[0]) / (self.quad_chunk_size / np.sqrt(self.num_chunks))
                filtered_near[:, 1] = filtered_near[:, 1] / (self.quad_chunk_size / np.sqrt(self.num_chunks))
                filtered_near[:, 2] = (filtered_near[:, 2] - z_range[0]) / (self.quad_chunk_size / np.sqrt(self.num_chunks))
                filtered_near = filtered_near * 2 - 1

                large_chunk_actual_num_near.append(min(self.num_sample_near, filtered_near.shape[0]))
                if filtered_near.shape[0] < self.num_sample_near:
                    short_samples = np.arange(filtered_near.shape[0])
                    expand_samples = np.random.choice(short_samples, size=self.num_sample_near - filtered_near.shape[0], replace=True)
                    near_surface_indices_samples = np.concatenate([short_samples, expand_samples])
                else:
                    near_surface_indices_samples = np.random.choice(filtered_near.shape[0], size=self.num_sample_near, replace=False)
                filtered_near = filtered_near[near_surface_indices_samples]
                filtered_near_occ = filtered_near_occ[near_surface_indices_samples]
                large_chunk_near_occs.append(filtered_near_occ)
                large_chunk_near_occ_queries.append(filtered_near)

        large_chunk_pcds = torch.stack(large_chunk_pcds).numpy().astype(np.float16)
        large_chunk_actual_num_pcds = np.array(large_chunk_actual_num_pcds).astype(np.int32)

        large_chunk_pos_occ_queries = torch.stack(large_chunk_pos_occ_queries).numpy().astype(np.float16)
        large_chunk_actual_num_pos = np.array(large_chunk_actual_num_pos).astype(np.int32)

        large_chunk_neg_occ_queries = torch.stack(large_chunk_neg_occ_queries).numpy().astype(np.float16)
        large_chunk_actual_num_neg = np.array(large_chunk_actual_num_neg).astype(np.int32)

        large_chunk_near_occ_queries = torch.stack(large_chunk_near_occ_queries).numpy().astype(np.float16)
        large_chunk_near_occs = torch.stack(large_chunk_near_occs).numpy().astype(np.int8)
        large_chunk_actual_num_near = np.array(large_chunk_actual_num_near).astype(np.int32)

        return {
            "chunk_path": chunk_path,
            "pcd": large_chunk_pcds,
            "num_actual_pcd": large_chunk_actual_num_pcds,

            "pos_surface_pos": large_chunk_pos_occ_queries,
            "num_actual_pos": large_chunk_actual_num_pos,

            "neg_surface_pos": large_chunk_neg_occ_queries,
            "num_actual_neg": large_chunk_actual_num_neg,

            "near_surface_pos": large_chunk_near_occ_queries,
            "near_surface_occ": large_chunk_near_occs,
            "num_actual_near": large_chunk_actual_num_near,

            "sample_x": chunk['sample_x'],
            "sample_y": chunk['sample_y']
        }

def _collate_fn(batch):
    batch_data = []
    for _, b in enumerate(batch):
        batch_data.append((b['chunk_path'], b['pcd'], b['num_actual_pcd'],
                           b['pos_surface_pos'], b['num_actual_pos'],
                           b['neg_surface_pos'], b['num_actual_neg'],
                           b['near_surface_pos'], b['near_surface_occ'], b['num_actual_near'],
                           b['sample_x'], b['sample_y']))
    return batch_data

batch_size = 10
num_workers = args.num_workers
quad_chunk_size = args.quad_chunk_size
dataset = ChunksDataset(glob.glob('data/{}_qcs{}/*/*.npz'.format(args.sample_json.split('/')[-1].split('.')[0], args.quad_chunk_size)), quad_chunk_size=quad_chunk_size)
chunks_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=False, collate_fn=_collate_fn)

with h5py.File('data/{}_qcs{}.h5'.format(args.sample_json.split('/')[-1].split('.')[0], args.quad_chunk_size), "w") as h5_file:
    h5_file.create_dataset("pcd", shape=(len(dataset), 4, 15000, 3), dtype=np.float16)
    h5_file.create_dataset("num_actual_pcd", shape=(len(dataset), 4), dtype=np.int32)

    h5_file.create_dataset("num_actual_pos", shape=(len(dataset), 4), dtype=np.int32)
    h5_file.create_dataset("pos_surface_pos", shape=(len(dataset), 4, 12500, 3), dtype=np.float16)

    h5_file.create_dataset("num_actual_neg", shape=(len(dataset), 4), dtype=np.int32)
    h5_file.create_dataset("neg_surface_pos", shape=(len(dataset), 4, 12500, 3), dtype=np.float16)

    h5_file.create_dataset("num_actual_near", shape=(len(dataset), 4), dtype=np.int32)
    h5_file.create_dataset("near_surface_pos", shape=(len(dataset), 4, 12500, 3), dtype=np.float16)
    h5_file.create_dataset("near_surface_occ", shape=(len(dataset), 4, 12500), dtype=np.int8)

    h5_file.create_dataset("sample_x", shape=(len(dataset)), dtype=np.int32)
    h5_file.create_dataset("sample_y", shape=(len(dataset)), dtype=np.int32)

    counter = 0
    idx_to_model = {}
    for batch in tqdm(chunks_dataloader):
        for data in batch:
            chunk_path, pcd, num_actual_pcd, pos_surface_pos, num_actual_pos, neg_surface_pos, num_actual_neg, near_surface_pos, near_surface_occ, num_actual_near, sample_x, sample_y = data
            model_id = chunk_path.split('/')[-2]
            idx_to_model[counter] = model_id

            h5_file["pcd"][counter] = pcd
            h5_file["num_actual_pcd"][counter] = num_actual_pcd

            h5_file["num_actual_pos"][counter] = num_actual_pos
            h5_file["pos_surface_pos"][counter] = pos_surface_pos

            h5_file["num_actual_neg"][counter] = num_actual_neg
            h5_file["neg_surface_pos"][counter] = neg_surface_pos

            h5_file["num_actual_near"][counter] = num_actual_near
            h5_file["near_surface_pos"][counter] = near_surface_pos
            h5_file["near_surface_occ"][counter] = near_surface_occ

            h5_file["sample_x"][counter] = sample_x
            h5_file["sample_y"][counter] = sample_y
            counter += 1

with open('data/{}_qcs{}.json'.format(args.sample_json.split('/')[-1].split('.')[0], args.quad_chunk_size), "w") as f:
    json.dump(idx_to_model, f, indent=4)

print('Finished processing data/{}_qcs{}.h5 and data/{}_qcs{}.json'.format(args.sample_json.split('/')[-1].split('.')[0], args.quad_chunk_size, args.sample_json.split('/')[-1].split('.')[0], args.quad_chunk_size))
