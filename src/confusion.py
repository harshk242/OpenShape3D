import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(model_dict, fig_size=(13,10), file_name="src/eval_data/mn_confusion.png",
                          save_fig=False):
    # modelnet_dict = torch.load("src/eval_data/modelnet_dict.pt")

    per_cat_acc = model_dict["per_cat_acc"]
    labels_all = model_dict["labels_all"]
    logits_all = model_dict["logits_all"]
    idx2category = model_dict["idx2category"]
    category2idx = model_dict["category2idx"]

    cat_list = list(idx2category.values())
    # print(cat_list.index("mantel")) #34
    # quit()
    
    print("labels_all: ", labels_all.shape)
    
    _, pred = logits_all.topk(1, 1, True, True)
    pred = pred.reshape(-1)
    
    true_pred = torch.argwhere(pred == labels_all).reshape(-1)
    false_pred = torch.argwhere(pred != labels_all).reshape(-1)
    
    res = torch.zeros(len(cat_list), len(cat_list))
    for val in true_pred:
        idx = val.item()
        res[labels_all[idx], pred[idx]] += 1
    
    for val in false_pred:
        idx = val.item()
        res[labels_all[idx], pred[idx]] += 1

    
    res_norm = F.softmax(res, dim=1)
    
    # result = []
    # for i in range(len(cat_list)):
    #     idx = torch.where(labels_all == i)[0]
    #     # print(logits_all[idx].sum(dim=0)/idx.shape[0])
    #     result.append(logits_all[idx].sum(dim=0)/idx.shape[0])   
    #     # print("idx: ", idx)
    #     # break
    # result = torch.stack(result)
    # res_norm = F.softmax(result, dim=1)

    df = pd.DataFrame(res_norm.numpy())

    plt.figure(figsize=fig_size)
    map = sns.heatmap(df, xticklabels=cat_list, yticklabels=cat_list, cmap="rocket_r")
    map.set_ylabel("True")
    map.set_xlabel("Predicted")
    # plt.imshow(df, cmap='hot', interpolation='nearest')
    if save_fig:
        plt.savefig(file_name)

    return res, res_norm

if __name__ == '__main__':
    modelnet_dict = torch.load("src/eval_data/modelnet_dict.pt")
    objaverse_dict = torch.load("src/eval_data/objaverse_dict.pt")

    mn_fig_size = (13,10)
    ov_fig_size = (200,200)

    mn_file_name = "src/eval_data/mn_confusion.png"
    ov_file_name = "src/eval_data/ov_confusion.png"

    mn_res, mn_res_norm = plot_confusion_matrix(modelnet_dict, fig_size=mn_fig_size, 
                                                file_name=mn_file_name, save_fig=False)
    ov_res, ov_res_norm = plot_confusion_matrix(objaverse_dict, fig_size=ov_fig_size, 
                                                file_name=ov_file_name, save_fig=False)

    torch.save(mn_res, "src/eval_data/mn_confusion.pt")
    torch.save(mn_res_norm, "src/eval_data/mn_confusion_norm.pt")

    torch.save(ov_res, "src/eval_data/ov_confusion.pt")
    torch.save(ov_res_norm, "src/eval_data/ov_confusion_norm.pt")

    # plot_confusion_matrix()
    # objaverse_confusion()

# def modelnet_confusion():
#     modelnet_dict = torch.load("src/eval_data/modelnet_dict.pt")

#     per_cat_acc = modelnet_dict["per_cat_acc"]
#     labels_all = modelnet_dict["labels_all"]
#     logits_all = modelnet_dict["logits_all"]
#     idx2category = modelnet_dict["idx2category"]
#     category2idx = modelnet_dict["category2idx"]

#     # mn_per_cat_acc = torch.load("src/eval_data/mn_per_cat_acc.pt")
#     # mn_labels_all = torch.load("src/eval_data/mn_labels_all.pt")
#     # mn_logits_all = torch.load("src/eval_data/mn_logits_all.pt")
#     # idx2category = torch.load("src/eval_data/mn_idx2category.pt")

#     # for t in labels_all:
#     #     print(t.shape)
#     cat_list = list(idx2category.values())
#     # print("idx2category: ", cat_list)
#     # quit()
#     # print("mn_per_cat_acc: ", mn_per_cat_acc)
#     # print("logits_all: ", logits_all)

#     _, pred = logits_all.topk(1, 1, True, True)
#     pred = pred.reshape(-1)
#     # print("pred: ", pred)
#     # print("labels_all: ", labels_all)
#     # print("indices: ", pred.shape)

#     true_pred = torch.argwhere(pred == labels_all).reshape(-1)
#     false_pred = torch.argwhere(pred != labels_all).reshape(-1)

#     res = torch.zeros(40, 40)
#     for val in true_pred:
#         idx = val.item()
#         res[labels_all[idx], pred[idx]] += 1
    
#     for val in false_pred:
#         idx = val.item()
#         res[labels_all[idx], pred[idx]] += 1
#         # print(labels_all[idx], pred[idx])

#     # res[39, 10] = 90
#     res_norm = F.softmax(res, dim=1)
#     # print(len(labels_all))
#     # result = []
#     # for i in range(len(cat_list)):
#     #     idx = torch.where(labels_all == i)[0]
#     #     # print(logits_all[idx].sum(dim=0)/idx.shape[0])
#     #     result.append(logits_all[idx].sum(dim=0)/idx.shape[0])   
#     #     # print("idx: ", idx)
#     #     # break
#     # result = torch.stack(result)
#     # res_norm = F.softmax(result, dim=1)
#     # print("result: ", res_norm.shape)

#     df = pd.DataFrame(res_norm.numpy())
#     # df["labels"] = cat_list[:30]

#     plt.figure(figsize=(13,10))
#     map = sns.heatmap(df, xticklabels=cat_list, yticklabels=cat_list, cmap="rocket_r")
#     map.set_ylabel("True")
#     map.set_xlabel("Predicted")
#     # plt.imshow(df, cmap='hot', interpolation='nearest')
#     plt.savefig("src/eval_data/mn_confusion.png")

# def objaverse_confusion():
#     objaverse_dict = torch.load("src/eval_data/objaverse_dict.pt")
#     per_cat_acc = objaverse_dict["per_cat_acc"]
#     labels_all = objaverse_dict["labels_all"]
#     logits_all = objaverse_dict["logits_all"]
#     idx2category = objaverse_dict["idx2category"]
#     category2idx = objaverse_dict["category2idx"]

#     cat_list = list(idx2category.values())


#     result = []
#     for i in range(len(idx2category)):
#         idx = torch.where(labels_all == i)[0]
#         result.append(logits_all[idx].sum(dim=0)/idx.shape[0])
    
#     result = torch.stack(result)
#     res_norm = F.softmax(result, dim=1)

#     df = pd.DataFrame(res_norm.numpy())
#     # df["labels"] = cat_list[:30]

#     plt.figure(figsize=(200,200))
#     map = sns.heatmap(df, xticklabels=cat_list, yticklabels=cat_list, cmap="rocket_r")
#     map.set_ylabel("True")
#     map.set_xlabel("Predicted")
#     # plt.imshow(df, cmap='hot', interpolation='nearest')
#     plt.savefig("src/eval_data/ov_confusion.png")


# if __name__ == '__main__':
#     modelnet_confusion()
    # objaverse_confusion()