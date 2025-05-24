# 🪡Nui(縫い)Scene43 Dataset

## NuiScene: Exploring Efficient Generation of Unbounded Outdoor Scenes

[Han-Hung Lee](https://hanhung.github.io/), [Qinghong Han](https://sulley.cc/), [Angel X. Chang](https://angelxuanchang.github.io/)

**[Paper](https://arxiv.org/abs/2503.16375)** | **[Project Page](https://3dlg-hcvc.github.io/NuiScene/)** | **[Dataset Page](https://3dlg-hcvc.github.io/NuiScene43-Dataset/)** | **[Model Code](https://github.com/3dlg-hcvc/NuiScene)**

## Install Environment

```
# create and activate the conda environment
conda create -n sasu python=3.10
conda activate sasu

# install PyTorch
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# install Python libraries
pip install -r requirements.txt

# install diffusers
pip install --upgrade diffusers[torch]

# install torch-cluster
conda install pytorch-cluster -c pyg

# install positional encoding
pip install positional-encodings[pytorch]
```

## Download and Visualize Dataset

1. Download dataset
```
python utils/download_data.py
```
2. Visualize scene. Will save a ***scene_id.obj*** and ***scene_id_maps.png***. Object file saves the marching cubes of the occupancy grid of the scene and the image file shows the sample maps used to sample chunks from the scene.
```
python utils/vis_data.py --scene_id=857673bc44c8411ca8aca7cab3be7091

# Some scenes will have scene_id_0~scene_id_3. These scenes were too large for our occupancy conversion script. So we split the scene into smaller scene blocks for occupancy conversion.
python utils/vis_data.py --scene_id=5f1822bbb40c43b097c4c98ecc697ed2_0
```

## Sample h5 for Training

## Citation

```
@misc{lee2025nuisceneexploringefficientgeneration,
      title={NuiScene: Exploring Efficient Generation of Unbounded Outdoor Scenes}, 
      author={Han-Hung Lee and Qinghong Han and Angel X. Chang},
      year={2025},
      eprint={2503.16375},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.16375}, 
}
```

## Acknowledgements

This work was funded by a CIFAR AI Chair, an NSERC Discovery grant, and a CFI/BCKDF JELF grant. We thank Jiayi Liu, and Xingguang Yan for helpful suggestions on improving the paper.
