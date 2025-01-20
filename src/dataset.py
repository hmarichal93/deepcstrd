from urudendro.labelme import load_ring_shapes
from urudendro.image import load_image, write_image

import numpy as np
import cv2
import random

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image, ImageDraw

import scipy.ndimage


def elastic_deformation(image, mask=None, alpha=15, sigma=3, random_state=None):
    """
    Apply elastic deformation to an image (and mask if provided).

    Args:
        image (np.ndarray): The input image to be deformed (H, W, C) or (H, W).
        mask (np.ndarray): The corresponding mask to deform (optional, H, W).
        alpha (float): Scaling factor for displacement fields (higher = stronger deformation).
        sigma (float): Standard deviation of the Gaussian filter (smoothness of deformation).
        random_state (np.random.RandomState): Random state for reproducibility.

    Returns:
        tuple: Deformed image and mask (if mask is provided).
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    image = np.array(image)  # Ensure NumPy array
    shape = image.shape[:2]  # Only use the spatial dimensions (H, W)

    # Generate random displacement fields
    dx = random_state.rand(*shape) * 2 - 1  # Random field in range [-1, 1]
    dy = random_state.rand(*shape) * 2 - 1
    dx = scipy.ndimage.gaussian_filter(dx, sigma=sigma, mode="constant", cval=0) * alpha
    dy = scipy.ndimage.gaussian_filter(dy, sigma=sigma, mode="constant", cval=0) * alpha

    # Create meshgrid for coordinates
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.array([
        np.clip(y + dy, 0, shape[0] - 1).flatten(),  # Y-coordinate deformation
        np.clip(x + dx, 0, shape[1] - 1).flatten()   # X-coordinate deformation
    ])

    # Apply deformation to the image
    if image.ndim == 3:  # Multichannel image
        deformed_image = np.zeros_like(image)
        for i in range(image.shape[2]):  # Apply deformation to each channel independently
            deformed_image[..., i] = scipy.ndimage.map_coordinates(
                image[..., i], indices, order=1, mode="reflect"
            ).reshape(shape)
    else:  # Single-channel image
        deformed_image = scipy.ndimage.map_coordinates(
            image, indices, order=1, mode="reflect"
        ).reshape(shape)

    if mask is not None:
        # Apply deformation to the mask
        deformed_mask = scipy.ndimage.map_coordinates(
            mask, indices, order=0, mode="reflect"
        ).reshape(shape)
        return deformed_image, deformed_mask

    return deformed_image

def unet_check(image):
    "Image must be divisible by 32"
    h, w = image.shape[:2]
    if h % 32 != 0 or w % 32 != 0:
        return False
    return True
def create_tiles_with_labels(image, mask, tile_size, overlap):
    H, W = image.shape[:2]
    stride = int(tile_size * (1 - overlap))
    image_tiles, mask_tiles = [], []
    for i in range(0, H, stride):
        for j in range(0, W, stride):
            i_max = min(i + tile_size, H - 1)
            if i_max == H-1:
                i = H - tile_size

            j_max = min(j + tile_size, W-1)
            if j_max == W - 1:
                j = W - tile_size

            image_tile = image[i:i_max, j:j_max]
            mask_tile = mask[i:i_max, j:j_max]
            # if mask_tile.sum() == 0 or not unet_check(image_tile):
            #     continue # Skip images with no ring

            if image_tile.ndim == 2:
                image_tile = cv2.cvtColor(image_tile, cv2.COLOR_GRAY2BGR)

            if image_tile.shape[0] < tile_size or image_tile.shape[1] < tile_size:
                #tile is inhomogeeous
                aux = np.zeros((tile_size, tile_size, 3), dtype=np.uint8) + 255
                aux[:image_tile.shape[0], :image_tile.shape[1]] = image_tile
                image_tile = aux

                if mask_tile.ndim == 2:
                    aux = np.zeros((tile_size, tile_size), dtype=np.uint8)
                else:
                    aux = np.zeros((tile_size, tile_size, 3), dtype=np.uint8) + 255
                aux[:mask_tile.shape[0], :mask_tile.shape[1]] = mask_tile
                mask_tile = aux

            image_tiles.append(image_tile)
            mask_tiles.append(mask_tile)
    return np.array(image_tiles), np.array(mask_tiles)

def overlay_images(background, overlay, alpha=0.5, beta=0.5, gamma=0):
    # Blend the images
    blended = cv2.addWeighted(background, alpha, overlay.astype(np.uint8), beta, gamma)
    return blended

def from_tiles_to_image(tiles, tile_size, original_image, overlap=0.2, output_dir=None, img=None):
    # Reconstruct the image from the tiles
    max_value = np.max(tiles)
    image = np.zeros((original_image.shape[0] , original_image.shape[1]), dtype=tiles.dtype)
    H,W = original_image.shape[:2]
    stride = int(tile_size * (1 - overlap))
    image_tiles, mask_tiles = [], []
    idx = 0
    for i in range(0, H, stride):
        for j in range(0, W , stride):

            i_max = min(i + tile_size, H - 1)
            if i_max == H - 1:
                i = H - tile_size
            j_max = min(j + tile_size, W - 1)
            if j_max == W - 1:
                j = W - tile_size

            image[i:i_max, j:j_max]+= tiles[idx][:(i_max-i),:(j_max-j)]
            if output_dir:
                debug_image = img.copy()
                pred_tile = np.zeros(((i_max-i), (j_max-j), 3), dtype=np.uint8)
                pred_tile[:,:,0] = tiles[idx][:(i_max-i),:(j_max-j)]*255
                #pred_tile = np.stack([tiles[idx]]*3, axis=2)*255
                debug_image[i:i_max, j:j_max] = overlay_images(debug_image[i:i_max, j:j_max], pred_tile,
                                                               alpha=0.5, beta=0.5, gamma=0)
                #overlay the tile on the image

                write_image(f"{output_dir}/tile_{idx}.png", debug_image)

            idx += 1
    return np.clip(image,0,max_value)

def generate_random_vector_multinomial_numpy(size, percentage):
    """
    Generate a random binary vector using multinomial distribution (NumPy).

    Args:
        size (int): Length of the vector.
        percentage (float): Probability of 1s (between 0 and 100).

    Returns:
        np.ndarray: A binary vector with 1s based on the given probability.
    """
    num_ones = int(size * (percentage / 100.0))  # Number of 1s to include
    indices = np.random.choice(size, num_ones, replace=False)  # Random indices for 1s
    vector = np.zeros(size, dtype=int)
    vector[indices] = 1
    return vector

class OverlapTileDataset(Dataset):
    def __init__(self, dataset_dir: Path, tile_size: int, overlap: int, tiles: bool = True, augmentation: bool = False,
                 augment_percentage=20,
                 debug: bool = True):
        self.images_dir = dataset_dir / "images/segmented"
        self.annotations_dir = dataset_dir / "annotations/labelme/images/"
        self.mask_dir = dataset_dir / "masks"
        self.augment_percentage = augment_percentage
        if debug:
            self.tiles_dir = dataset_dir / "tiles"
            if self.tiles_dir.exists():
                import os
                os.system(f"rm -r {self.tiles_dir}")
            self.tiles_images_dir = self.tiles_dir / "images"
            self.tiles_masks_dir = self.tiles_dir / "masks"
            self.tiles_dir.mkdir(parents=True, exist_ok=True)
            self.tiles_images_dir.mkdir(parents=True, exist_ok=True)
            self.tiles_masks_dir.mkdir(parents=True, exist_ok=True)


        self.images, self.labels = self.load_data(tiles, tile_size, overlap, augmentation, debug)

    def augment_data(self, images, masks, percentage, horizontal_flip=True, vertical_flip=True, occlusions=True,
                     elastic_deformation=True, rotation=True):
        """
        Augment data. Add rotation 45, 90, 135, 180.
        Add Flip
        Add occlusions.
        """
        augmented_images = []
        augmented_masks = []
        #generate samples of 0 and 1 to decide if apply augmentation
        vector = generate_random_vector_multinomial_numpy(len(images), percentage)
        idx = -1
        for img, mask in zip(images, masks):
            idx += 1
            if vector[idx] == 0:
                continue
            img_pil = Image.fromarray(img)  # Convert to PIL for transformations
            mask_pil = Image.fromarray(mask.astype(float) * 255)  # Scale mask to 0-255 for transformations

            # Apply augmentations (rotate, flip, and occlusions)
            aug_img, aug_mask = self.apply_augmentations(img_pil, mask_pil, vertical_flip, horizontal_flip, occlusions,
                                                         elastic_deformation, rotation)

            # Convert back to NumPy arrays
            augmented_images.append(np.array(aug_img))
            augmented_masks.append((np.array(aug_mask) > 127).astype(np.uint8))  # Threshold mask back to binary

        return augmented_images, augmented_masks

    def apply_augmentations(self, image, mask, vertical_flip=True, horizontal_flip=True, occlusions=True, elastic=True,
                            rotation=True):
        """
        Apply specific augmentations to an image and its mask.
        """
        # Random rotations
        if rotation:
            rotations = [0, 45, 90, 135, 180, 225, 270, 315]
            angle = random.choice(rotations)
            image = image.rotate(angle)
            mask = mask.rotate(angle)

        # Random horizontal flip
        if random.random() > 0.5 and horizontal_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() > 0.5 and vertical_flip:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Add random occlusions
        if occlusions:
            image = self.add_occlusions(image)
            mask = self.add_occlusions(mask, mask_occlusion=True)

        if elastic:
            image, mask = elastic_deformation(image, mask, alpha=50, sigma=5)

        return image, mask

    def add_occlusions(self, image, mask_occlusion=False):
        """
        Add random occlusions to the image or mask.
        """
        draw = ImageDraw.Draw(image)
        width, height = image.size

        # Add 1-3 random rectangles

        for _ in range(random.randint(1, 3)):
            width_rectangle = random.randint(0, width // 2)
            height_rectangle = random.randint(0, height // 2)
            x1 = random.randint(0, width -1)
            y1 = random.randint(0, height -1)
            x2 = np.minimum(x1 + width_rectangle, width-1)
            y2 = np.minimum(y1 + height_rectangle, height-1)
            # For masks, use black; for images, use random colors
            color = 0 if mask_occlusion else tuple(random.randint(0, 255) for _ in range(3))
            draw.rectangle([x1, y1, x2, y2], fill=color)

        return image

    def load_data(self, tiles, tile_size, overlap, augmentation, debug):
        images, masks = self.load_images_and_masks(mask_dir=self.mask_dir)


        l_images, l_labels = [], []
        if tiles:
            for image, mask in zip(images, masks):
                tiles, labels = create_tiles_with_labels(image, mask, tile_size, overlap)
                l_images.extend(tiles)
                l_labels.extend(labels)

        else:
            l_images = images
            l_labels = masks

        if augmentation:
            # images_aug, masks_aug = self.augment_data(l_images, l_labels, self.augment_percentage, elastic_deformation=False)
            # l_images.extend(images_aug)
            # l_labels.extend(masks_aug)
            #
            # images_aug, masks_aug = self.augment_data(l_images, l_labels, self.augment_percentage, horizontal_flip=False,
            #                                           vertical_flip=False,  elastic_deformation=False)
            # l_images.extend(images_aug)
            # l_labels.extend(masks_aug)

            images_aug, masks_aug = self.augment_data(l_images, l_labels, self.augment_percentage, horizontal_flip=False,
                                                      vertical_flip=False,  elastic_deformation=True, occlusions=False,
                                                      rotation=False)
            l_images.extend(images_aug)
            l_labels.extend(masks_aug)

        if debug:
            counter = 0
            for t, l in zip(l_images, l_labels):
                write_image(self.tiles_images_dir / f"{counter}.png", t)
                write_image(self.tiles_masks_dir / f"{counter}.png", (l * 255).astype(np.uint))
                counter += 1
        return l_images, l_labels


    def load_images_and_masks(self, mask_dir=None):
        if mask_dir is not None:
            mask_dir.mkdir(parents=True, exist_ok=True)
        annotations = list(self.annotations_dir.glob("*.json"))
        l_mask = []
        l_images = []
        for ann in annotations:
            try:
                img_path = next(self.images_dir.rglob(f"{ann.stem}*"))
            except StopIteration:
                continue
            image = load_image(img_path)

            #convert to RGB
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            mask = self.annotation_to_mask(ann, image)
            y, x = np.where(mask > 0)
            y_min, y_max = np.min(y), np.max(y)
            offset = 10
            y_min = np.maximum(0, y_min - offset)
            y_max = np.minimum(image.shape[0]-1, y_max + offset)
            x_min, x_max = np.min(x), np.max(x)
            x_min = np.maximum(0, x_min - offset)
            x_max = np.minimum(image.shape[1]-1, x_max + offset)
            #crop image and mask with an offset of 10 pixels
            image = image[y_min:y_max, x_min:x_max]
            mask = mask[y_min:y_max, x_min:x_max]
            if mask_dir is not None:
                mask_path = mask_dir / f"{ann.stem}.png"
                write_image(mask_path, mask)

            mask = np.where(mask > 0, 1, 0)
            l_mask.append(mask)
            l_images.append(image)


        return l_images, l_mask

    def annotation_to_mask(self, annotation, img, boundaries_thickness = 3):
        """
        Transform annotation to mask
        :param annotation: annotation path
        :return: mask
        """
        l_rings = load_ring_shapes(annotation)
        # 1.0 create mask
        boundaries_mask = np.zeros(img.shape[:2], dtype=np.int8)
        # 2.0 fill mask
        for i, ring in enumerate(l_rings):
            cv2.polylines(boundaries_mask, [ring.points.astype(np.int32)], isClosed=True,
                          color=255, thickness=boundaries_thickness)

        return boundaries_mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def main( dataset_dir = "/data/maestria/datasets/Pinus_Taeda/PinusTaedaV1", tile_size=512, overlap=0.20):
    dataset = OverlapTileDataset( Path(dataset_dir), tile_size=tile_size, overlap=overlap)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return

def split_dataset(dataset_root:Path, val_size=0.2, test_size=0.2):
    images_dir = dataset_root / "images/segmented"
    annotations_dir = dataset_root / "annotations/labelme/images"
    annotations_dir_path = [ann_path for ann_path in annotations_dir.glob("*.json")]

    # suffle the dataset
    np.random.seed(42)
    lenght = len(annotations_dir_path)
    # generate a list from 0 to lenght
    l_indexes = np.arange(lenght)
    np.random.shuffle(l_indexes)
    # split the dataset
    val_index = int(lenght * val_size)
    test_index = int(lenght * test_size)
    train_index = lenght - val_index - test_index

    train_images_index = l_indexes[:train_index]
    val_images_index = l_indexes[train_index:train_index + val_index]
    test_images_index = l_indexes[train_index + val_index:]
    ##
    generate_dataset_folder(dataset_root / "train", dataset_root,
                            [annotations_dir_path[i].stem for i in train_images_index])

    generate_dataset_folder(dataset_root / "val", dataset_root,
                            [annotations_dir_path[i].stem for i in val_images_index])

    generate_dataset_folder(dataset_root / "test", dataset_root,
                            [annotations_dir_path[i].stem for i in test_images_index])



    return

from typing import List
import os
def generate_dataset_folder(folder_name:Path, dataset_root:Path, samples_list:List[str]):
    """
    Generate a dataset folder with the samples in samples_list
    """
    images_dir = dataset_root / "images/segmented"
    annotations_dir = dataset_root / "annotations/labelme/images"

    images_dir_dest = folder_name / "images/segmented"
    annotations_dir_dest = folder_name / "annotations/labelme/images"

    images_dir_dest.mkdir(parents=True, exist_ok=True)
    annotations_dir_dest.mkdir(parents=True, exist_ok=True)
    for sample in samples_list:
        image_path = next(images_dir.rglob(f"*{sample}*"))
        annotation_path = annotations_dir / f"{sample}.json"
        image_dest = images_dir_dest / image_path.name
        annotation_dest = annotations_dir_dest / annotation_path.name

        os.system(f"cp {image_path} {image_dest}")
        os.system(f"cp {annotation_path} {annotation_dest}")

    return

if __name__ == "__main__":
    main()