from urudendro.image import load_image, write_image
from urudendro.drawing import Drawing, Color
from urudendro.labelme import AL_LateWood_EarlyWood
from pathlib import Path
from shapely.geometry import Polygon

def main(input="input/F02c"):
    input_image = f"{input}.png"
    im_in = load_image(input_image)
    ann_path = f"{input}.json"
    ann = AL_LateWood_EarlyWood(ann_path, None)
    shapes = ann.read()
    im_in_debug = im_in.copy()
    for s in shapes:
        poly = Polygon(s.points)
        Drawing.curve(poly.exterior.coords, im_in_debug, Color.blue, thickness=1)

    write_image(f"{input}_out_1.png", im_in_debug)
    im_in_debug = im_in.copy()

    ann = AL_LateWood_EarlyWood(ann_path, None)
    shapes = ann.read()
    for s in shapes:
        poly = Polygon(s.points)
        Drawing.curve(poly.exterior.coords, im_in_debug, Color.blue, thickness=2)

    write_image(f"{input}_out_2.png", im_in_debug)

    return 0


def generate_inspection(input = './input/C17-2', type='deep'):
    input_image = f"{input}.png"
    im_in = load_image(input_image)
    ann_path = f"{input}_{type}.json"
    ann = AL_LateWood_EarlyWood(ann_path, None)
    shapes = ann.read()
    im_in_debug = im_in.copy()
    for s in shapes:
        poly = Polygon(s.points)
        Drawing.curve(poly.exterior.coords, im_in_debug, Color.blue, thickness=3)

    write_image(f"{input}_{type}.png", im_in_debug)


if __name__ == "__main__":
    #main()
    generate_inspection()