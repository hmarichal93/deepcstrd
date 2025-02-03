from urudendro.image import load_image
from pathlib import Path

from deep_cstrd.deep_tree_ring_detection import DeepTreeRingDetection
from cross_section_tree_ring_detection.cross_section_tree_ring_detection import save_config, saving_results

def main(args):
    save_config(args, args.root, args.output_dir)
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input image {args.input} not found")

    im_in = load_image(args.input)

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    res = DeepTreeRingDetection(im_in, args.cy, args.cx, args.sigma, args.th_low, args.th_high, args.hsize, args.wsize,
                            args.edge_th, args.nr, args.min_chain_length, args.weights_path, args.total_rotations,
                            args.debug, args.input, args.output_dir, args.tile_size)

    saving_results(res, args.output_dir, args.save_imgs)

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False,default="input/F02c.png")
    parser.add_argument("--weights_path", type=str, required=False,
                        default="/home/henry/Documents/repo/fing/cores_tree_ring_detection/src/runs/pinus_v1/latest_model.pth")
    parser.add_argument("--cy", type=int, required=False, default=1264)
    parser.add_argument("--cx", type=int, required=False, default=1204)
    parser.add_argument("--root", type=str, required=False, default="./")
    parser.add_argument("--output_dir", type=str, required=False, default="./output/F02c")
    parser.add_argument("--save_imgs", type=int, required=False, default=1)
    parser.add_argument("--sigma", type=float, required=False,default=3)
    parser.add_argument("--nr", type=int, required=False,default=360)
    parser.add_argument("--hsize", type=int, required=False, default=0)
    parser.add_argument("--wsize", type=int, required=False, default=0)
    parser.add_argument("--edge_th", type=int, required=False, default=30)
    parser.add_argument("--th_high", type=int, required=False, default=20)
    parser.add_argument("--th_low", type=int, required=False, default=5)
    parser.add_argument("--min_chain_length", type=int, required=False, default=2)
    parser.add_argument("--total_rotations", type=int, required=False, default=4)
    parser.add_argument('--tile_size', type=int, required=False, default=0)
    parser.add_argument("--debug", type=int, required=False)

    args = parser.parse_args()
    main(args)