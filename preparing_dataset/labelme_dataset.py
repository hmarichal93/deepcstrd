"""
Script for removing the background of the Salix Glauca dataset and generated a file call dataset_ipol.csv where
it is stored the information pith center of each image.
"""

from urudendro.remove_salient_object import remove_salient_object
from urudendro.labelme import AL_LateWood_EarlyWood, resize_annotations, LabelmeObject
from urudendro.image import  load_image, resize_image_using_pil_lib, write_image

from deep_cstrd.dataset import padding_image

from pathlib import Path

import numpy as np
import os
import cv2

def get_minimum_bounding_box(image, offset=50):
    """Minimum bounding box that include the disk"""
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    y, x = np.where(img_gray < 255)
    y_min, y_max = np.min(y), np.max(y)
    y_min = np.maximum(0, y_min - offset)
    y_max = np.minimum(image.shape[0] - 1, y_max + offset)
    x_min, x_max = np.min(x), np.max(x)
    x_min = np.maximum(0, x_min - offset)
    x_max = np.minimum(image.shape[1] - 1, x_max + offset)

    return y_min, y_max, x_min, x_max
def crop_image(image_path, annotation_path, output_image_path, output_annotation_path, offset=50):
    image = load_image(image_path)
    y_min, y_max, x_min, x_max = get_minimum_bounding_box(image, offset)
    new_image = image[y_min:y_max, x_min:x_max]
    write_image(output_image_path, new_image)

    ann = AL_LateWood_EarlyWood(annotation_path, output_annotation_path, image_path=None)
    shapes = ann.read()
    for shape in shapes:
        shape.points = (shape.points - [y_min, x_min])
        shape.points = shape.points[:, ::-1]

    object = LabelmeObject()
    object.from_memory(shapes=shapes, imagePath=str(Path(image_path).name))
    json_content = object.to_dict()
    ann.write(json_content)
    return

from shapely.geometry import Polygon
from urudendro.drawing import Drawing
def draw_annotations_over_image(image, shapes):
    for shape in shapes:
        points = shape.points
        poly = Polygon(points)
        image = Drawing.curve(poly.exterior.coords, image, color=(255, 0, 0), thickness=2)

    return image

def generate_inspection(root_database = "/data/maestria/resultados/deep_cstrd_datasets/salix_glauca_2500", resize=True, hsize=1504, wsize=1504,
         remove_background = False, output_dir= "/data/maestria/resultados/deep_cstrd_datasets/salix_glauca_2500"):
    images_dir = Path(root_database) / "images/segmented"
    annotations_dir = Path(root_database) / "annotations/labelme/images"
    segmented_dir = Path(root_database) / "images/segmented"
    segmented_dir.mkdir(exist_ok=True, parents=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    annotations_output_dir = output_dir / "inspection"
    annotations_output_dir.mkdir(exist_ok=True, parents=True)


    for img_path in images_dir.glob("*"):
        if img_path.is_dir():
            continue
        img_name = img_path.stem
        print(img_name)
        ann_output_path = annotations_dir / f"{img_name}.json"
        ann = AL_LateWood_EarlyWood(ann_output_path, None)
        shapes = ann.read()
        img = load_image(str(img_path))
        img = draw_annotations_over_image(img, shapes)
        write_image(str(annotations_output_dir / f"{img_name}.png"), img)

    return



def main(root_database = "/data/maestria/resultados/deep_cstrd_datasets/salix_glauca", resize=True, hsize=1504, wsize=1504,
         remove_background = False, output_dir= "/data/maestria/resultados/deep_cstrd_datasets/salix_glauca_1504"):
    images_dir = Path(root_database) / "images/segmented"
    annotations_dir = Path(root_database) / "annotations/labelme/images"
    segmented_dir = Path(root_database) / "images/segmented"
    segmented_dir.mkdir(exist_ok=True, parents=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    annotations_output_dir = output_dir / "annotations/labelme/images"
    annotations_output_dir.mkdir(exist_ok=True, parents=True)
    segmented_output_dir = output_dir / "images/segmented"
    segmented_output_dir.mkdir(exist_ok=True, parents=True)
    resized_dir = segmented_output_dir.parent / "resized"
    resized_dir.mkdir(exist_ok=True, parents=True)

    metadata_filename = output_dir / 'dataset_ipol.csv'

    metadata = open(metadata_filename, "w")
    metadata.write("Imagen,cx,cy\n")

    for img_path in images_dir.glob("*"):
        if img_path.is_dir():
            continue
        img_name = img_path.stem
        print(img_name)
        ##crop image
        img_output_path = segmented_output_dir / f"{img_name}.png"
        img_output_path = segmented_output_dir / f"{img_name}.jpg"
        ann_output_path = annotations_output_dir / f"{img_name}.json" if resize else annotations_dir / f"{img_name}.json"
        ann_path = annotations_dir / f"{img_name}.json"
        crop_image(img_path, ann_path, img_output_path, ann_output_path)
        ###

        if not img_output_path.exists() and remove_background:
            img_output_path = segmented_dir / f"{img_name}.jpg"
            if not img_output_path.exists():
                remove_salient_object(str(img_path), str(img_output_path))

        image_resized = False
        if resize:
            img_array = load_image(str(img_output_path))
            if img_array.shape[2] == 4:
                #transform to rgb
                import cv2
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
            #if highest dimension is lower thant the desired dimension, then the image is not resized
            img_resized_path = resized_dir / f"{img_name}.png"
            #if not (img_array.shape[0] < hsize and img_array.shape[1] < wsize):
            img_r = resize_image_using_pil_lib(img_array, hsize, wsize, str(img_resized_path))

            write_image(str(img_resized_path), img_r)
            image_resized = True


        else:
            img_array = load_image(str(img_output_path))
            #if highest dimension is lower thant the desired dimension, then the image is not resized
            img_resized_path = resized_dir / f"{img_name}.png"
            write_image(str(img_resized_path), img_array)



        ann_aux_path = None
        if image_resized:
            ann_aux_path = f"/tmp/{img_name}.json"
            os.system(f"cp {ann_output_path} {ann_aux_path}")
            ann_output_path = resize_annotations(str(img_output_path), str(img_resized_path), str(ann_aux_path))

        #continue
        if resize and (img_r.shape[0] % 32 != 0 or img_r.shape[1] % 32 != 0):
            img_r = padding_image(img_r)
            write_image(str(img_resized_path), img_r)

        ann = AL_LateWood_EarlyWood(ann_output_path, None, image_path= str(img_resized_path))
        shapes = ann.read()
        pith_ann = shapes[0]
        pith_pixel = pith_ann.points.mean(axis=0).astype(int)
        cy,cx = pith_pixel
        metadata.write(f"{img_name},{cx},{cy}\n")
        if ann_aux_path is not None:
            ann_aux_path = annotations_output_dir / f"{img_name}.json"
            os.system(f"cp -rf {ann_output_path} {ann_aux_path}")
            #os.system(f"rm {ann_path}")


    #remove images in segmented directory and copy the resized images

    os.system(f"rm -rf {segmented_output_dir}/*")
    os.system(f"cp -fr {resized_dir}/* {segmented_output_dir}")
    metadata.close()
    return


if __name__ == "__main__":
    main()
    #generate_inspection()
