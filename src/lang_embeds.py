import torch
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans, DBSCAN
import torch.nn.functional as F


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_feat_path = "meta_data/lvis_cat_name_pt_feat.npy"
clip_cat_feat = np.load(clip_feat_path, allow_pickle=True)
objaverse_dict = torch.load("src/eval_data/objaverse_dict.pt")
ov_category2idx = objaverse_dict["category2idx"]
ov_idx2category = objaverse_dict["idx2category"]
ov_category2idx = objaverse_dict["category2idx"] 

# clip_cat_feat = F.normalize(clip_cat_feat, dim=-1)
clip_cat_feat = clip_cat_feat / norm(clip_cat_feat, axis=1, keepdims=True)

eps = 0.61855
db_scan = DBSCAN(eps=eps, min_samples=2).fit(clip_cat_feat)
dbscan_labels = db_scan.labels_

obj_idx = ov_category2idx["sherbert"]
val = dbscan_labels[obj_idx]
idx = np.argwhere(dbscan_labels == val)

obj_list = []
for i in idx:
    obj_list.append(ov_idx2category[i.item()])

print("obj_list: ", len(obj_list))
print("obj_list: ", obj_list)

print("eps: ", eps)
print("dbscan: ", np.unique(db_scan.labels_))

