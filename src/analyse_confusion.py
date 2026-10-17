import torch
import numpy as np
import matplotlib.pyplot as plt


def top_confusion():
    topx = 40
    topy = 5
    # modelnet_dict = torch.load("src/eval_data/modelnet_dict.pt")
    # idx2category = modelnet_dict["idx2category"]
    objaverse_dict = torch.load("src/eval_data/objaverse_dict.pt")
    ov_idx2category = objaverse_dict["idx2category"]
    ov_xyz = torch.load("src/eval_data/ov_xyz.pt")

    # mn_res_norm = torch.load("src/eval_data/mn_confusion_norm.pt")
    ov_res_norm = torch.load("src/eval_data/ov_confusion_norm.pt")


    ov_res = ov_res_norm.fill_diagonal_(0)
    row_sum = torch.sum(ov_res, dim=1)
    # print("row_sum: ", row_sum.shape) # 1156 categories

    sorted_idx = torch.argsort(row_sum, descending=True)
    print("sorted: ", sorted_idx)
    sort_cat = [ov_idx2category[i.item()] for i in sorted_idx[:topx]]

    vodka = sorted_idx[0]


    for i in sorted_idx[:topx]:
        topy_pred = torch.argsort(ov_res[i], descending=True)[:topy]
        topy_cat = [ov_idx2category[i.item()] for i in topy_pred]
        print(ov_idx2category[i.item()], ": ", topy_cat)


def plot_pointcloud(obj_name):
    objaverse_dict = torch.load("src/eval_data/objaverse_dict.pt")
    ov_xyz = torch.load("src/eval_data/ov_xyz.pt")
    ov_idx2category = objaverse_dict["idx2category"]
    ov_category2idx = objaverse_dict["category2idx"]
    ov_labels_all = objaverse_dict["labels_all"]

    idx = ov_category2idx[obj_name]
    obj_instance = np.random.choice(ov_labels_all[ov_labels_all == idx])
    obj_xyz = ov_xyz[obj_instance]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(obj_xyz[:,0], obj_xyz[:,1], obj_xyz[:,2], s=0.1)
    plt.savefig(f"src/eval_data/{obj_name}_{obj_instance}.png")

def explore_features():
    xyz = torch.load("src/eval_data/ov_xyz.pt")
    rgb = torch.load("src/eval_data/ov_rgb.pt")
    features = torch.load("src/eval_data/ov_features.pt")
    print("rgb: ", rgb)
    print("xyz: ", xyz.shape)

explore_features()
# plot_pointcloud("vodka")



def calc_var():

    mn_res = torch.load("src/eval_data/mn_confusion.pt")
    mn_res_norm = torch.load("src/eval_data/mn_confusion_norm.pt")


    modelnet_dict = torch.load("src/eval_data/modelnet_dict.pt")
    idx2category = modelnet_dict["idx2category"]
    labels_all = modelnet_dict["labels_all"]
    data_xyz = modelnet_dict["data_xyz"]
    xyz_var = torch.var(data_xyz, dim=1)


    print("xyz_var: ", xyz_var.shape)

    class_var_list = []
    for i in range(len(idx2category)):
        class_var = xyz_var[labels_all == i].mean(dim=0)
        class_var_list.append(class_var)

    class_var_list = torch.stack(class_var_list)

    for i, var in enumerate(class_var_list):
        if len(idx2category[i]) < 5:
            print(idx2category[i], ": \t\t", var)
        else: 
            print(idx2category[i], ": \t", var)
        

