import os
import argparse
import pickle
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from ruamel.yaml import safe_load
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils.helpers import dir_exists, remove_files


def data_process(data_path, name, patch_size, stride, mode):
    save_path = os.path.join(data_path, f"{mode}_pro")
    dir_exists(save_path)
    remove_files(save_path)
    if name == "DRIVE":
        img_path = os.path.join(data_path, mode, "images")
        gt_path = os.path.join(data_path, mode, "1st_manual")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "CHASEDB1":
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "STARE":
        img_path = os.path.join(data_path, "stare-images")
        gt_path = os.path.join(data_path, "labels-ah")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "DCA1":
        data_path = os.path.join(data_path, "Database_134_Angiograms")
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "CHUAC":
        img_path = os.path.join(data_path, "Original")
        gt_path = os.path.join(data_path, "Photoshop")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "roads":
        img_path = os.path.join(data_path, "tiles_up")
        mask_path = os.path.join(data_path, "masks_26apr")
        file_list = list(sorted(os.listdir(img_path)))

    img_list = []
    gt_list = []
    for i, file in enumerate(file_list):
        if name == "DRIVE":
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))

        elif name == "CHASEDB1":
            if len(file) == 13:
                if mode == "training" and int(file[6:8]) <= 10:
                    img = Image.open(os.path.join(data_path, file))
                    gt = Image.open(os.path.join(
                        data_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[6:8]) > 10:
                    img = Image.open(os.path.join(data_path, file))
                    gt = Image.open(os.path.join(
                        data_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
        elif name == "DCA1":
            if len(file) <= 7:
                if mode == "training" and int(file[:-4]) <= 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[:-4]) > 100:
                    img = cv2.imread(os.path.join(data_path, file), 0)
                    gt = cv2.imread(os.path.join(
                        data_path, file[:-4] + '_gt.pgm'), 0)
                    gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
        elif name == "CHUAC":
            if mode == "training" and int(file[:-4]) <= 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                if int(file[:-4]) <= 17 and int(file[:-4]) >= 11:
                    tail = "PNG"
                else:
                    tail = "png"
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok."+tail), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
            elif mode == "test" and int(file[:-4]) > 20:
                img = cv2.imread(os.path.join(img_path, file), 0)
                gt = cv2.imread(os.path.join(
                    gt_path, "angio"+file[:-4] + "ok.png"), 0)
                gt = np.where(gt >= 100, 255, 0).astype(np.uint8)
                img = cv2.resize(
                    img, (512, 512), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(f"save_picture/{i}img.png", img)
                cv2.imwrite(f"save_picture/{i}gt.png", gt)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
        elif name == "STARE":
            if not file.endswith("gz"):
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(gt_path, file[0:6] + '.ah.ppm'))
                cv2.imwrite(f"save_picture/{i}img.png", np.array(img))
                cv2.imwrite(f"save_picture/{i}gt.png", np.array(gt))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))

        if name == "roads":
            if not file.endswith("_band1.tif"):
                continue  # Process only band1 files once (each tile)

            # Build tile_base from file name
            tile_base = file[:-10]  # removes '_band1.tif' part

            # ---- Load 4 bands ----
            bands = []
            for band_idx in range(1, 5):
                band_filename = f"{tile_base}_band{band_idx}.tif"
                band_img = Image.open(os.path.join(img_path, band_filename)).convert('L')
                band_array = np.array(band_img, dtype=np.float32) / 255.0
                bands.append(band_array)

            image = np.stack(bands, axis=0)  # Shape (4, H, W)
            image_tensor = torch.from_numpy(image).float()

            # ---- Load and union 4 masks ----
            masks = []
            for band_idx in range(1, 5):
                mask_filename = f"{tile_base}_band{band_idx}.png"
                mask_full_path = os.path.join(mask_path, mask_filename)

                if os.path.exists(mask_full_path):
                    mask_img = Image.open(mask_full_path).convert('L')
                    mask_array = np.array(mask_img, dtype=np.uint8)
                    mask_array = (mask_array > 0).astype(np.uint8)
                    masks.append(mask_array)

            if len(masks) == 0:
                raise FileNotFoundError(f"No masks found for tile {tile_base}")

            union_mask = np.any(masks, axis=0).astype(np.uint8)
            mask_tensor = torch.from_numpy(union_mask).unsqueeze(0).float()  # [1, H, W]

            # ---- Pad to 600x600 if needed ----
            C, H, W = image_tensor.shape
            TARGET_H, TARGET_W = 600, 600
            pad_h = TARGET_H - H
            pad_w = TARGET_W - W

            if pad_h > 0 or pad_w > 0:
                image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=0)
                mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), value=0)

            img_list.append(image_tensor)
            gt_list.append(mask_tensor)

    img_list = normalization(img_list)
    if mode == "training":
        img_patch = get_patch(img_list, patch_size, stride)
        gt_patch = get_patch(gt_list, patch_size, stride)
        save_patch(img_patch, save_path, "img_patch", name)
        save_patch(gt_patch, save_path, "gt_patch", name)
    elif mode == "test":
        if name != "CHUAC":
            img_list = get_square(img_list, name)
            gt_list = get_square(gt_list, name)
        save_each_image(img_list, save_path, "img", name)
        save_each_image(gt_list, save_path, "gt", name)


def get_square(img_list, name):
    img_s = []
    if name == "DRIVE":
        shape = 592
    elif name == "CHASEDB1":
        shape = 1008
    elif name == "DCA1":
        shape = 320
    _, h, w = img_list[0].shape
    pad = nn.ConstantPad2d((0, shape-w, 0, shape-h), 0)
    for i in range(len(img_list)):
        img = pad(img_list[i])
        img_s.append(img)

    return img_s


def get_patch(imgs_list, patch_size, stride):
    image_list = []
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    for sub1 in imgs_list:
        image = F.pad(sub1, (0, pad_w, 0, pad_h), "constant", 0)
        image = image.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride).permute(1, 2, 0, 3, 4)
        image = image.contiguous().view(
            image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)
        for sub2 in image:
            image_list.append(sub2)
    return image_list


def save_patch(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def save_each_image(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')


def normalization(imgs_list):
    # imgs = torch.cat(imgs_list, dim=0)
    # mean = torch.mean(imgs)
    # std = torch.std(imgs)
    imgs = torch.cat(imgs_list, dim=0)  # [N, C, H, W]
    mean = imgs.mean(dim=[0,2,3])  # mean over all patches per channel
    std = imgs.std(dim=[0,2,3])
    normal_list = []
    for i in imgs_list:
        # n = Normalize([mean], [std])(i)
        n = Normalize(mean.tolist(), std.tolist())(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
    return normal_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="datasets/DRIVE", type=str,
                        help='the path of dataset',required=True)
    parser.add_argument('-dn', '--dataset_name', default="DRIVE", type=str,
                        help='the name of dataset',choices=['DRIVE','CHASEDB1','STARE','CHUAC','DCA1','road'],required=True)
    parser.add_argument('-ps', '--patch_size', default=48,
                        help='the size of patch for image partition')
    parser.add_argument('-s', '--stride', default=6,
                        help='the stride of image partition')
    args = parser.parse_args()
    with open('config.yaml', encoding='utf-8') as file:
        CFG = safe_load(file)  # 为列表类型

    data_process(args.dataset_path, args.dataset_name,
                 args.patch_size, args.stride, "training")
    data_process(args.dataset_path, args.dataset_name,
                 args.patch_size, args.stride, "test")
