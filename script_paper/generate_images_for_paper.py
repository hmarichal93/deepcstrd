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


def visualize_chains_over_image(chain_list=[], img=None, filename=None, devernay=None, filter=None):

    import matplotlib.pyplot as plt
    figsize = (10, 10)
    plt.figure(figsize=figsize)
    #from bgr to rgb
    img = img[...,::-1]
    plt.imshow(img)
    for chain in chain_list:
        x = chain[:,1]
        y = chain[:,0]
        x = x.tolist() + [x[0]]
        y = y.tolist() + [y[0]]
        plt.plot(x, y, 'b', linewidth=1)


    plt.tight_layout()
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def generate_inspection(input = './input/c08d', type='inbd'):
    input_image = f"{input}.png"
    im_in = load_image(input_image)
    ann_path = f"{input}_{type}.json"
    ann = AL_LateWood_EarlyWood(ann_path, None)
    shapes = ann.read()
    im_in_debug = im_in.copy()
    chain_list = []
    for s in shapes:
        poly = Polygon(s.points)
        chain_list.append(s.points)
        Drawing.curve(poly.exterior.coords, im_in_debug, Color.blue, thickness=2)

    write_image(f"{input}_{type}.png", im_in_debug)
    visualize_chains_over_image(chain_list, im_in, f"{input}_{type}_chains.png")


if __name__ == "__main__":
    #main()
    generate_inspection()