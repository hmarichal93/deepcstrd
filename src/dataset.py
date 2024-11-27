from urudendro.labelme import load_ring_shapes
from urudendro.image import load_image, write_image

import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def unet_check(image):
    "Image must be divisible by 32"
    h, w = image.shape[:2]
    if h % 32 != 0 or w % 32 != 0:
        return False
    return True
def create_tiles_with_labels(image, mask, tile_size, overlap):

    stride = tile_size - overlap
    image_tiles, mask_tiles = [], []
    for i in range(0, image.shape[0] - tile_size + 1, stride):
        for j in range(0, image.shape[1] - tile_size + 1, stride):
            image_tile = image[i:i+tile_size, j:j+tile_size]
            mask_tile = mask[i:i+tile_size, j:j+tile_size]
            if mask_tile.sum() == 0 or not unet_check(image_tile):
                continue # Skip images with no ring
            #number of channels of image_tile
            if image_tile.shape[2] != 3:
                print(f"Image shape: {image_tile.shape}")
            image_tiles.append(image_tile)
            mask_tiles.append(mask_tile)
    return np.array(image_tiles), np.array(mask_tiles)


class OverlapTileDataset(Dataset):
    def __init__(self, dataset_dir: Path, tile_size: int, overlap: int, tiles: bool = True, augmentation: bool = False,
                 debug: bool = True):
        self.images_dir = dataset_dir / "images/segmented"
        self.annotations_dir = dataset_dir / "annotations/mean_gt/"
        self.mask_dir = dataset_dir / "masks"
        if debug:
            self.tiles_dir = dataset_dir / "tiles"
            self.tiles_images_dir = self.tiles_dir / "images"
            self.tiles_masks_dir = self.tiles_dir / "masks"
            self.tiles_dir.mkdir(parents=True, exist_ok=True)
            self.tiles_images_dir.mkdir(parents=True, exist_ok=True)
            self.tiles_masks_dir.mkdir(parents=True, exist_ok=True)

        self.images, self.labels = self.load_data(tiles, tile_size, overlap, augmentation,  debug)

    def augment_data(self, images, masks):
        return images, masks
    def load_data(self, tiles, tile_size, overlap, augmentation, debug):
        images, masks = self.load_images_and_masks(mask_dir=self.mask_dir)
        if augmentation:
            images, masks = self.augment_data(images, masks)

        l_images, l_labels = [], []
        if tiles:
            for image, mask in zip(images, masks):
                tiles, labels = create_tiles_with_labels(image, mask, tile_size, overlap)
                l_images.extend(tiles)
                l_labels.extend(labels)
                if debug:
                    counter = 0
                    for t, l in zip(tiles, labels):
                        write_image(self.tiles_images_dir / f"{len(l_images) + counter}.png", t)
                        write_image(self.tiles_masks_dir / f"{len(l_images) + counter}.png", l)
                        counter += 1
        else:
            l_images = images
            l_labels = masks

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

def main( dataset_dir = "/data/maestria/datasets/Pinus_Taeda/PinusTaedaV1", tile_size=512, overlap=128):
    dataset = OverlapTileDataset( Path(dataset_dir), tile_size=tile_size, overlap=overlap)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    return

if __name__ == "__main__":
    main()