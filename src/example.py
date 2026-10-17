import os
import numpy as np
import open3d as o3d
import random
import torch
import sys
from param import parse_args
import models
from utils.data import normalize_pc
from utils.misc import load_config
from huggingface_hub import hf_hub_download
from collections import OrderedDict
import open_clip
import re
from PIL import Image
import torch.nn.functional as F
import collections

device = torch.device("cuda")

def batched_coordinates(coords, dtype=torch.int32, device=None):
    r"""Create a `ME.SparseTensor` coordinates from a sequence of coordinates

    Given a list of either numpy or pytorch tensor coordinates, return the
    batched coordinates suitable for `ME.SparseTensor`.

    Args:
        :attr:`coords` (a sequence of `torch.Tensor` or `numpy.ndarray`): a
        list of coordinates.

        :attr:`dtype`: torch data type of the return tensor. torch.int32 by default.

    Returns:
        :attr:`batched_coordindates` (`torch.Tensor`): a batched coordinates.

    .. warning::

       From v0.4, the batch index will be prepended before all coordinates.

    """
    assert isinstance(
        coords, collections.abc.Sequence
    ), "The coordinates must be a sequence."
    assert np.array(
        [cs.ndim == 2 for cs in coords]
    ).all(), "All coordinates must be in a 2D array."
    D = np.unique(np.array([cs.shape[1] for cs in coords]))
    D = D[0]

    # Create a batched coordinates
    N = np.array([len(cs) for cs in coords]).sum()
    bcoords = torch.zeros((N, D + 1), dtype=dtype, device=device)  # uninitialized

    s = 0
    for b, cs in enumerate(coords):
        if dtype == torch.int32:
            if isinstance(cs, np.ndarray):
                cs = torch.from_numpy(np.floor(cs))
            elif not (
                isinstance(cs, torch.IntTensor) or isinstance(cs, torch.LongTensor)
            ):
                cs = cs.floor()

            cs = cs.int()
        else:
            if isinstance(cs, np.ndarray):
                cs = torch.from_numpy(cs)

        cn = len(cs)
        # BATCH_FIRST:
        bcoords[s : s + cn, 1:] = cs
        bcoords[s : s + cn, 0] = b
        s += cn
    return bcoords

def load_ply(file_name, num_points=10000, y_up=True):
    pcd = o3d.io.read_point_cloud(file_name)
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    n = xyz.shape[0]
    if n != num_points:
        idx = random.sample(range(n), num_points)
        xyz = xyz[idx]
        rgb = rgb[idx]
    if y_up:
        # swap y and z axis
        xyz[:, [1, 2]] = xyz[:, [2, 1]]
    xyz = normalize_pc(xyz)
    if rgb is None:
        rgb = np.ones_like(rgb) * 0.4
    features = np.concatenate([xyz, rgb], axis=1)
    xyz = torch.from_numpy(xyz).type(torch.float32)
    features = torch.from_numpy(features).type(torch.float32)
    # print("xyz: ", xyz.shape)
    # print("features: ", features.shape)
    # print("batch: ", xyz.unsqueeze(0).shape)
    # xyz:  torch.Size([10000, 3])
    # features:  torch.Size([10000, 6])
    # xyz = batched_coordinates([xyz], dtype=torch.float32)
    # print("xyz: ", xyz.shape)
    return xyz.unsqueeze(0), features.unsqueeze(0)
    # return xyz, features
    return ME.utils.batched_coordinates([xyz], dtype=torch.float32), features

def load_model(config, model_name="OpenShape/openshape-pointbert-vitg14-rgb"):
    print("Loading model: ", model_name)
    model = models.make(config).cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in checkpoint['state_dict'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
    model.load_state_dict(model_dict)
    # model.to(device)
    return model

@torch.no_grad()
def extract_text_feat(texts, clip_model,):
    text_tokens = open_clip.tokenizer.tokenize(texts).cuda()
    return clip_model.encode_text(text_tokens)

@torch.no_grad()
def extract_image_feat(images, clip_model, clip_preprocess):
    image_tensors = [clip_preprocess(image) for image in images]
    image_tensors = torch.stack(image_tensors, dim=0).float().cuda()
    image_features = clip_model.encode_image(image_tensors)
    image_features = image_features.reshape((-1, image_features.shape[-1]))
    return image_features

print("loading OpenShape model...")
cli_args, extras = parse_args(sys.argv[1:])
config = load_config("src/configs/train.yaml", cli_args = vars(cli_args), extra_args = extras)
# model_name = "OpenShape/openshape-pointbert-no-lvis"
# model_name = "OpenShape/openshape-pointbert-shapenet"
model = load_model(config)
model.to(device)
# new_model = nn.Sequential(*list(model.children())[:-2])
# for name in list(model.children())[:1]:
#     print("name: ", name)
    # print("param: ", param)

# print("model: ", model)
model.eval()

cache_dir = os.environ.get("OPEN_CLIP_CACHE_DIR")
print("loading OpenCLIP model...")
open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', 
                                            pretrained='laion2b_s39b_b160k', cache_dir=cache_dir)
open_clip_model.cuda().eval()
# print("clip_model: ", open_clip_model)

print("extracting 3D shape feature...")
xyz, feat = load_ply("demo/owl.ply")
xyz = xyz.to(device)
feat = feat.to(device)
shape_feat = model(xyz, feat, device='cuda', quantization_size=config.model.voxel_size) 

print("extracting text features...")
texts = ["owl", "chicken", "penguin"]
text_feat = extract_text_feat(texts, open_clip_model)
print("texts: ", texts)
print("3D-text similarity: ", F.normalize(shape_feat, dim=1) @ F.normalize(text_feat, dim=1).T)

print("extracting image features...")
image_files = ["demo/a.jpg", "demo/b.jpg", "demo/c.jpg"]
images = [Image.open(f).convert("RGB") for f in image_files]
image_feat = extract_image_feat(images, open_clip_model, open_clip_preprocess)
print("image files: ", image_files)
print("3D-image similarity: ", F.normalize(shape_feat, dim=1) @ F.normalize(image_feat, dim=1).T)

