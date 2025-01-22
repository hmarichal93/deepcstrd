import numpy as np
import os
from pathlib import Path
from PIL import Image

from urudendro.image import  load_image, write_image

from deep_cstrd.dataset import overlay_images

def resize_image_using_pil_lib(im_in: np.array, height_output: object, width_output: object, keep_ratio= True,
                               flag = Image.Resampling.LANCZOS) -> np.ndarray:
    """
    Resize image using PIL library.
    @param im_in: input image
    @param height_output: output image height_output
    @param width_output: output image width_output
    @return: matrix with the resized image
    """

    pil_img = Image.fromarray(im_in)
    # Image.ANTIALIAS is deprecated, PIL recommends using Reampling.LANCZOS
    #flag = Image.ANTIALIAS
    if keep_ratio:
        aspect_ratio = pil_img.height / pil_img.width
        if pil_img.width > pil_img.height:
            height_output = int(width_output * aspect_ratio)
        else:
            width_output = int(height_output / aspect_ratio)

    pil_img = pil_img.resize((width_output, height_output), flag)
    im_r = np.array(pil_img)
    if flag == Image.Resampling.NEAREST:
        #thresholding
        im_r[im_r < 128] = 0



    return im_r
def generate_root_directories(output_dir, ann_dir, images_dir, RING_BOUNDARY_VALUE=0, PITH_VALUE=1,
                              resize=True, hsize=1500, wsize=1500, debug=True):
    images_output_dir = output_dir / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir = output_dir / "mask"
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_filename = output_dir / 'dataset_ipol.csv'
    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    metadata = open(metadata_filename, "w")
    metadata.write("Imagen,cx,cy\n")

    for image_path in images_dir.rglob("*.jpg"):
        ann_path = ann_dir / f"{Path(image_path).stem}.tiff"
        if not ann_path.exists():
            raise ValueError(f"Annotation {ann_path} not found")

        # convertir ann_path a una imagen binaria
        ann = load_image(str(ann_path))
        mask = np.zeros_like(ann).astype(np.uint8)
        mask[ann == RING_BOUNDARY_VALUE] = 255

        if resize:
            mask = resize_image_using_pil_lib(mask, hsize, wsize, flag=Image.Resampling.NEAREST)



        # get pith pixel
        cyy, cxx = np.where(ann == PITH_VALUE)
        cy = int(cyy.mean().round())
        cx = int(cxx.mean().round())
        if resize:
            cy = int(cy * mask.shape[0] / ann.shape[0])
            cx = int(cx * mask.shape[1] / ann.shape[1])

        # generate labelme json
        # TODO

        # save image
        image_output_path = images_output_dir / f"{Path(image_path).stem}.png"
        if resize:
            img = load_image(str(image_path))
            img = resize_image_using_pil_lib(img, hsize, wsize)
            write_image(image_output_path, img)
        else:
            os.system(f"cp {image_path} {image_output_path}")

        # save labelme json
        # TODO

        # save mask
        mask_output_path = mask_output_dir / f"{Path(image_path).stem}.png"
        write_image(mask_output_path, mask)

        img_name = Path(image_path).stem
        metadata.write(f"{img_name},{cx},{cy}\n")

        if debug:
            #overlay mask over image
            img = load_image(str(image_output_path))
            overlay = np.zeros_like(img)
            overlay[:, :, 0] = mask
            overlay = overlay_images(img, overlay, alpha=0.5, beta=0.5, gamma=0)
            write_image(debug_dir / f"{img_name}.png", overlay)

    metadata.close()
    return mask_output_dir, images_output_dir

def load_txt_file(file_path):
    with open(file_path, "r") as f:
        return f.readlines()

def get_sample_subset(train_images_path):
    train_images_files = load_txt_file(train_images_path)
    train_samples = [Path(image).stem for image in train_images_files]
    return train_samples


def generate_subset(root_database: Path, output_dir: Path, images_output_dir: Path, mask_output_dir: Path, subset: str):
    train_images_path = root_database / f"{subset}_inputimages.txt"
    train_samples = get_sample_subset(train_images_path)

    train_folder = output_dir / subset
    train_folder.mkdir(parents=True, exist_ok=True)

    images_output = train_folder / "images/segmented"
    images_output.mkdir(parents=True, exist_ok=True)
    mask_output = train_folder / "masks"
    mask_output.mkdir(parents=True, exist_ok=True)
    for sample in train_samples:
        os.system(f"cp {images_output_dir}/{sample}.png {images_output}/")
        os.system(f"cp {mask_output_dir}/{sample}.png {mask_output}/")

    return train_folder
def main(root_database = "/data/maestria/datasets/Candice_inbd_1500/", output_dir = "./output", resize=True, hsize=1500,
         wsize=1500):
    root_database = Path(root_database)
    ann_dir = root_database / "annotations"
    images_dir = root_database / "inputimages"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_output_dir, images_output_dir = generate_root_directories(output_dir, ann_dir, images_dir)

    #generate train, val, test split
    _ = generate_subset(root_database, output_dir, images_output_dir, mask_output_dir, "train")
    test_folder = generate_subset(root_database, output_dir, images_output_dir, mask_output_dir, "test")
    val_folder = output_dir / "val"
    val_folder.mkdir(parents=True, exist_ok=True)
    os.system(f"cp -r {test_folder}/* {val_folder}/")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_database", type=str, default="/data/maestria/datasets/INBD/EH")
    parser.add_argument("--output_dir", type=str, default="/data/maestria/resultados/deep_cstrd/EH_1500")

    args = parser.parse_args()
    main(root_database= args.root_database, output_dir=  args.output_dir)