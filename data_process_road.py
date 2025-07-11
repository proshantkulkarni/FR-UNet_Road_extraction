import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def data_process(data_path, patch_size, stride, mode):
    # Paths
    tiles_dir = os.path.join(data_path, "tiles_up")
    masks_dir = os.path.join(data_path, "masks_26apr")
    save_path = os.path.join(data_path, f"{mode}_pro")

    # Ensure directories
    os.makedirs(save_path, exist_ok=True)
    save_dir = 'save_picture'
    os.makedirs(save_dir, exist_ok=True)

    # Find all tile base names
    tile_bases = sorted({f.rsplit('_band', 1)[0] for f in os.listdir(tiles_dir)})

    print(f"\nProcessing {len(tile_bases)} tiles...\n")

    for tile_base in tile_bases:
        bands = []
        masks = []

        for band_idx in range(1, 5):
            band_filename = f"{tile_base}_band{band_idx}.tif"
            mask_filename = f"{tile_base}_band{band_idx}.png"

            band_path = os.path.join(tiles_dir, band_filename)
            mask_path = os.path.join(masks_dir, mask_filename)

            if not os.path.exists(band_path):
                raise FileNotFoundError(f"Missing band image: {band_filename}")

            # Load band
            band_img = Image.open(band_path).convert('L')
            band_array = np.array(band_img, dtype=np.float32) / 255.0
            bands.append(band_array)

            # Load mask if exists
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert('L')
                mask_array = (np.array(mask_img, dtype=np.uint8) > 0).astype(np.uint8)
                masks.append(mask_array)

        # Stack bands
        image_tensor = torch.from_numpy(np.stack(bands, axis=0)).float()

        # Union of masks
        if masks:
            union_mask = np.any(masks, axis=0).astype(np.uint8)
        else:
            union_mask = np.zeros_like(bands[0], dtype=np.uint8)

        mask_tensor = torch.from_numpy(union_mask).unsqueeze(0).float()

        # Padding to (600,600)
        TARGET_H, TARGET_W = 600, 600
        _, H, W = image_tensor.shape
        pad_h = TARGET_H - H
        pad_w = TARGET_W - W
        if pad_h > 0 or pad_w > 0:
            image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=0)
            mask_tensor = F.pad(mask_tensor, (0, pad_w, 0, pad_h), value=0)

        # --- Normalize each image individually ---
        mean = torch.mean(image_tensor)
        std = torch.std(image_tensor)
        image_tensor = Normalize([mean], [std])(image_tensor)
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

        # --- Visualization immediately after loading each tile ---
        image_np = image_tensor.numpy()
        mask_np = mask_tensor.squeeze(0).numpy()

        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.1, hspace=0.15)

        band_titles = ["Band 1", "Band 2", "Band 3", "Band 4"]
        for i in range(4):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            ax.imshow(image_np[i], cmap='gray')
            ax.set_title(band_titles[i], fontsize=10)
            ax.axis('off')

        ax5 = fig.add_subplot(gs[0, 2])
        ax5.imshow(mask_np, cmap='gray')
        ax5.set_title("Union Mask", fontsize=10)
        ax5.axis('off')

        plt.suptitle(f"{tile_base}", fontsize=12)
        plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)
        plt.savefig(os.path.join(save_dir, f"{tile_base}.png"), dpi=150)
        plt.close('all')  # Critical to free memory

        # --- Save immediately ---
        if mode == "training":
            img_patches = get_patch([image_tensor], patch_size, stride)
            gt_patches = get_patch([mask_tensor], patch_size, stride)
            save_patch(img_patches, save_path, "img_patch")
            save_patch(gt_patches, save_path, "gt_patch")
        elif mode == "test":
            save_each_image([image_tensor], save_path, "img")
            save_each_image([mask_tensor], save_path, "gt")

    print(f"\n✅ Completed saving visualizations and processed data for {len(tile_bases)} tiles.\n")

def get_patch(imgs_list, patch_size, stride):
    image_list = []
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    for img in imgs_list:
        img_padded = F.pad(img, (0, pad_w, 0, pad_h), "constant", 0)
        patches = img_padded.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(-1, img.shape[0], patch_size, patch_size)
        image_list.extend(patches)
    return image_list

def save_patch(imgs_list, path, type_name):
    for i, img in enumerate(imgs_list):
        filename = f'{type_name}_{i}.pkl'
        with open(os.path.join(path, filename), 'wb') as file:
            pickle.dump(np.array(img), file)
        print(f"Saved {filename}")

def save_each_image(imgs_list, path, type_name):
    for i, img in enumerate(imgs_list):
        filename = f'{type_name}_{i}.pkl'
        with open(os.path.join(path, filename), 'wb') as file:
            pickle.dump(np.array(img), file)
        print(f"Saved {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', type=str, required=True,
                        help='Path to dataset folder containing tiles_up and masks_26apr')
    parser.add_argument('-ps', '--patch_size', type=int, default=48,
                        help='Patch size for training')
    parser.add_argument('-s', '--stride', type=int, default=6,
                        help='Stride size for patch extraction')
    args = parser.parse_args()

    data_process(args.dataset_path, args.patch_size, args.stride, mode="training")
    data_process(args.dataset_path, args.patch_size, args.stride, mode="test")
