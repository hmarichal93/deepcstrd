"""
Script for removing the background of the Salix Glauca dataset and generated a file call dataset_ipol.csv where
it is stored the information pith center of each image.
"""

from urudendro.remove_salient_object import remove_salient_object
from urudendro.labelme import AL_LateWood_EarlyWood, resize_annotations
from urudendro.image import  load_image, resize_image_using_pil_lib, write_image

from pathlib import Path

import numpy as np
import os

def main(root_database = "/data/maestria/resultados/deep_cstrd/pinus_v1", resize=True, hsize=1500, wsize=1500):
    images_dir = Path(root_database) / "images/original"
    annotations_dir = Path(root_database) / "annotations/labelme/images"
    segmented_dir = Path(root_database) / "images/segmented"
    segmented_dir.mkdir(exist_ok=True, parents=True)
    resized_dir =  segmented_dir.parent / "resized"
    resized_dir.mkdir(exist_ok=True, parents=True)
    metadata_filename = Path(root_database) / 'dataset_ipol.csv'
    metadata = open(metadata_filename, "w")
    metadata.write("Imagen,cx,cy\n")
    for img_path in images_dir.glob("*"):
        if img_path.is_dir():
            continue
        img_name = img_path.stem

        img_out = segmented_dir / f"{img_name}.png"
        if not img_out.exists():
            img_out = segmented_dir / f"{img_name}.jpg"
            if not img_out.exists():
                remove_salient_object(str(img_path), str(img_out))
        print(img_name)
        image_resized = False
        if resize:
            img_array = load_image(str(img_out))
            #if highest dimension is lower thant the desired dimension, then the image is not resized
            img_resized_path = resized_dir / f"{img_name}.png"
            if not (img_array.shape[0] < hsize and img_array.shape[1] < wsize):
                img_r = resize_image_using_pil_lib(img_array, hsize, wsize, str(img_resized_path))
                write_image(str(img_resized_path), img_r)
                image_resized = True
            else:
                write_image(str(img_resized_path), img_array)




        ann_path = annotations_dir / f"{img_name}.json"
        ann_aux_path = None
        if image_resized:
            ann_aux_path = f"/tmp/{img_name}.json"
            os.system(f"cp {ann_path} {ann_aux_path}")
            ann_path = resize_annotations(str(img_path), str(img_resized_path), str(ann_aux_path))


        ann = AL_LateWood_EarlyWood(ann_path, None, image_path=img_out)
        shapes = ann.read()
        pith_ann = shapes[0]
        pith_pixel = pith_ann.points.mean(axis=0).astype(int)
        cy,cx = pith_pixel
        metadata.write(f"{img_name},{cx},{cy}\n")
        if ann_aux_path is not None:
            ann_aux_path = annotations_dir / f"{img_name}.json"
            os.system(f"cp -rf {ann_path} {ann_aux_path}")
            os.system(f"rm {ann_path}")


    #remove images in segmented directory and copy the resized images

    os.system(f"rm -rf {segmented_dir}/*")
    os.system(f"cp -fr {resized_dir}/* {segmented_dir}")
    metadata.close()
    return


if __name__ == "__main__":
    main()