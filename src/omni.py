import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
import numpy as np
import random
import json

from utils.data import normalize_pc
from utils_func import calc_accuracy

def get_omni_dir(env_var):
    # OmniObject3D point cloud dirs, e.g. .../raw/point_clouds/hdf5_files/1024 and .../raw/point_clouds/ply_files
    path = os.environ.get(env_var)
    if path is None:
        raise RuntimeError(f"Set the {env_var} environment variable to the OmniObject3D point cloud directory.")
    return path

def load_omni_hdf5_data(num_points = 1024):
    omni_dict = np.load("../meta_data/omni_dict.npy", allow_pickle=True).item()
    idx2category = omni_dict['idx2category']
    category2idx = omni_dict['category2idx']
    keys = list(category2idx.keys())

    all_data = []
    all_label = []
    omni_hdf5_dir = get_omni_dir("OMNI_HDF5_DIR")
    omni_list = os.listdir(omni_hdf5_dir)

    for fname in omni_list:
        obj = fname.split('_')[0]
        if obj in keys:
            f = h5py.File(os.path.join(omni_hdf5_dir, obj+"_" + str(num_points) + ".hdf5"), 'r+')
            data = f['data'][0].astype('float64')
            idx = category2idx[obj]
            
            # label = f['label'][:].astype('int64')
            f.close()
            all_data.append([data])
            all_label.append([idx])
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    print("all_data: ", all_data.shape)
    print("all_label: ", all_label.shape)

    return all_data, all_label

def load_omni_ply_data(num_points = 1024):
    point_dir = os.path.join(get_omni_dir("OMNI_PLY_DIR"), str(num_points))
    omni_dict = np.load("../meta_data/omni_dict.npy", allow_pickle=True).item()
    idx2category = omni_dict['idx2category']
    category2idx = omni_dict['category2idx']
    keys = list(category2idx.keys())

    all_data = []
    all_label = []
    obj_list = os.listdir(point_dir)

    for obj in obj_list:
        if obj in keys:
            obj_label = category2idx[obj]
            sub_dir = os.path.join(point_dir, obj)
            instance_list = os.listdir(sub_dir)
            for ins in instance_list:
                ins_np = np.zeros((num_points, 3))
                ins_dir = os.path.join(sub_dir, ins)
                ply_file = os.path.join(ins_dir, "pcd_" + str(num_points) + ".ply")
                if os.path.exists(ply_file):
                    plydata = PlyData.read(ply_file)
                    ins_np[:, 0] = plydata['vertex']['x']
                    ins_np[:, 1] = plydata['vertex']['y']
                    ins_np[:, 2] = plydata['vertex']['z']
                    
                    all_data.append([ins_np])
                    all_label.append([obj_label])

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    print("all_data: ", all_data.shape)
    print("all_label: ", all_label.shape)

    return all_data, all_label
    
class OmniObject3D(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.objaverse_lvis.split, "r"))
        self.y_up = config.objaverse_lvis.y_up
        self.num_points = config.objaverse_lvis.num_points
        self.use_color = config.objaverse_lvis.use_color
        self.normalize = config.objaverse_lvis.normalize
        self.categories = sorted(np.unique([data['category'] for data in self.split]))
        self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}
        self.clip_cat_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)
        # logging.info("ObjaverseLVIS: %d samples" % (len(self.split)))
        # logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        path = self.split[index]['data_path'][1:]
        data = np.load(path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        
        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "rgb": torch.from_numpy(rgb).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "group": self.split[index]['group'],
            "name":  self.split[index]['uid'],
            "category": self.category2idx[self.split[index]["category"]],
        }

    def __len__(self):
        return len(self.split)
  

def objaverse_lvis_collate_fn(list_data):
    return {
        # "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "rgb_dense": torch.stack([data["rgb"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "group": [data["group"] for data in list_data],
        "name": [data["name"] for data in list_data],
        "category": torch.tensor([data["category"] for data in list_data], dtype = torch.int32),
    }


def make_objaverse_lvis(config):
    print("Loading ObjaverseLVIS...")
    return DataLoader(
        ObjaverseLVIS(config), \
        num_workers=config.objaverse_lvis.num_workers, \
        collate_fn=objaverse_lvis_collate_fn, \
        batch_size=config.objaverse_lvis.batch_size, \
        pin_memory = True, \
        shuffle=False 
    )

def test_objaverse_lvis(model, config, objaverse_lvis_loader, text_proj, device):
    model.eval()
    if config.training.use_text_proj:
        text_proj.eval()
    clip_text_feat = torch.from_numpy(objaverse_lvis_loader.dataset.clip_cat_feat).to(device)
    if config.training.use_text_proj:
        clip_text_feat = text_proj(clip_text_feat)
    per_cat_correct = torch.zeros(1156).to(device)
    per_cat_count = torch.zeros(1156).to(device)
    category2idx = objaverse_lvis_loader.dataset.category2idx
    idx2category = {v: k for k, v in category2idx.items()}
    
    logits_all = []
    labels_all = []
    data_xyz = []

    # pred_feat:  torch.Size([70, 1280])
    # clip_text_feat:  torch.Size([1156, 1280])
    # logits:  torch.Size([70, 1156])
    # lables:  torch.Size([70])

    with torch.no_grad():
        for data in objaverse_lvis_loader:

            data['xyz_dense'] = data['xyz_dense'].to(device)
            data['features_dense'] = data['features_dense'].to(device)
            
            pred_feat = model(data['xyz_dense'], data['features_dense'])
            logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
            labels = data['category'].to(device)
            data_xyz.append(data['xyz_dense'])
            logits_all.append(logits.detach())
            labels_all.append(labels)
            # calculate per class accuracy
            for i in torch.unique(labels):
                idx = (labels == i)
                if idx.sum() > 0:
                    per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
                    per_cat_count[i] += idx.sum()

    topk_acc, correct = calc_accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

    overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    per_cat_acc = per_cat_correct / per_cat_count

    print("Exporting ObjaverseLVIS results...")
    objaverse_dict = {}
    objaverse_dict["per_cat_acc"] = per_cat_acc.clone().detach().cpu()
    objaverse_dict["xyz"] = torch.cat(data_xyz).clone().detach().cpu()
    objaverse_dict["labels_all"] = torch.cat(labels_all).clone().detach().cpu()
    objaverse_dict["logits_all"] = torch.cat(logits_all).clone().detach().cpu()
    objaverse_dict["category2idx"] = category2idx
    objaverse_dict["idx2category"] = idx2category
    torch.save(objaverse_dict, "src/eval_data/objaverse_dict.pt")


    print('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
    print('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))


def xyz_objaverse_lvis(objaverse_lvis_loader, device):
    
    data_xyz = []
    with torch.no_grad():
        for data in objaverse_lvis_loader:
          
            data_xyz.append(data['xyz_dense'])

    torch.save(torch.cat(data_xyz).clone().detach().cpu(), "src/eval_data/ov_xyz.pt")

def rgb_objaverse_lvis(objaverse_lvis_loader, device):
    
    rgb = []
    with torch.no_grad():
        for data in objaverse_lvis_loader:
          
            rgb.append(data['rgb_dense'])

    torch.save(torch.cat(rgb).clone().detach().cpu(), "src/eval_data/ov_rgb.pt")

def features_objaverse_lvis(objaverse_lvis_loader, device):
    
    features = []
    with torch.no_grad():
        for data in objaverse_lvis_loader:
          
            features.append(data['features_dense'])

    torch.save(torch.cat(features).clone().detach().cpu(), "src/eval_data/ov_features.pt")