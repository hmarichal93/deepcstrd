import argparse, sys

from urudendro.image import load_image
from pathlib import Path

from deep_cstrd.deep_tree_ring_detection import DeepTreeRingDetection
from deep_cstrd.training import training, segmentation_model
from deep_cstrd.metrics_evaluation import evaluate

from cross_section_tree_ring_detection.cross_section_tree_ring_detection import save_config, saving_results


def inference(args):
    save_config(args, args.root, args.output_dir)
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input image {args.input} not found")

    im_in = load_image(args.input)

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    res = DeepTreeRingDetection(im_in, args.cy, args.cx, args.sigma, args.th_low, args.th_high, args.hsize, args.wsize,
                            args.edge_th, args.nr, args.min_chain_length, args.weights_path, args.total_rotations,
                            args.debug, args.input, args.output_dir, args.tile_size, args.prediction_map_threshold)

    saving_results(res, args.output_dir, args.save_imgs)

    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='subcommands', required=True)

    parser_inference = subparsers.add_parser('inference', help='Train a network')
    parser_inference.add_argument("--input", type=str, required=False,default="input/F02c.png")
    parser_inference.add_argument("--weights_path", type=str, required=False,
                        default="/home/henry/Documents/repo/fing/cores_tree_ring_detection/runs/20250130-221624/pinus_v1_1504_epochs_100_tile_256_batch_8_lr_0.001_resnet18_channels_3_thickness_3_augmentation/best_model.pth")
    parser_inference.add_argument("--cy", type=int, required=False, default=1264)
    parser_inference.add_argument("--cx", type=int, required=False, default=1204)
    parser_inference.add_argument("--root", type=str, required=False, default="./")
    parser_inference.add_argument("--output_dir", type=str, required=False, default="./output/F02c")
    parser_inference.add_argument("--save_imgs", type=int, required=False, default=1)
    parser_inference.add_argument("--sigma", type=float, required=False,default=3)
    parser_inference.add_argument("--nr", type=int, required=False,default=360)
    parser_inference.add_argument("--hsize", type=int, required=False, default=0)
    parser_inference.add_argument("--wsize", type=int, required=False, default=0)
    parser_inference.add_argument("--edge_th", type=int, required=False, default=30)
    parser_inference.add_argument("--th_high", type=int, required=False, default=20)
    parser_inference.add_argument("--th_low", type=int, required=False, default=5)
    parser_inference.add_argument("--min_chain_length", type=int, required=False, default=2)
    parser_inference.add_argument("--total_rotations", type=int, required=False, default=4)
    parser_inference.add_argument('--prediction_map_threshold', type=float, required=False, default=0.2)
    parser_inference.add_argument('--tile_size', type=int, required=False, default=256)
    parser_inference.add_argument("--debug", type=int, required=False)

    parser_inference.set_defaults(func=inference)

    parser_train = subparsers.add_parser('train', help='Train a network')
    parser_train.add_argument('--dataset_dir', type=str, default="/data/maestria/resultados/deep_cstrd/pinus_v1/",
                        help='Path to the dataset directory')
    parser_train.add_argument('--logs_dir', type=str, default="runs/pinus_v1_40_train_12_val")
    parser_train.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser_train.add_argument('--tile_size', type=int, default=256, help='Tile size')
    parser_train.add_argument('--step_size', type=int, default=20, help='Step size for the learning rate scheduler')
    parser_train.add_argument('--number_of_epochs', type=int, default=40, help='Number of epochs')
    parser_train.add_argument('--overlap', type=float, default=0.1, help='Overlap between tiles')
    parser_train.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser_train.add_argument('--loss', type=int, default=0, help='Loss function. 0 dice loss, 1 BCE loss')
    parser_train.add_argument('--encoder', type=str, default="resnet34", help='Encoder to use')
    parser_train.add_argument('--boundary_thickness', type=int, default=3, help='Mask boundary thickness')
    # parser.add_argument('--encoder', type=str, default="mobilenet_v2", help='Encoder to use')
    parser_train.add_argument('--input_channels', type=int, default=3, help='Number of input channels')
    # load rest of parameter from config file
    parser_train.add_argument("--config", type=str, default="config.json", help="Path to the config file")
    parser_train.add_argument("--augmentation", type=bool, default=False, help="Apply augmentation to the dataset")
    parser_train.add_argument("--model_type", type=int, default=segmentation_model.UNET, help="Type of model to use")
    parser_train.add_argument("--debug", type=bool, default=True, help="Debug mode")
    parser_train.set_defaults(func=training)

    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate the tree ring detection pipeline')
    parser_evaluate.add_argument("--dataset_dir", type=str, required=False,
                                 default="/data/maestria/resultados/deep_cstrd_datasets_train/pinus_v2_1504/test")
    parser_evaluate.add_argument("--results_path", type=str, required=False,
                                 default="/data/maestria/resultados/deep_cstrd_inbd/pinus_v2_1504/inference/inbd_results/2025-01-30_19h41m28s_INBD_100e_a6.3__/inbd_urudendro_labels/")
    parser_evaluate.set_defaults(func=evaluate)

    args = parser.parse_args(sys.argv[1:] or ['--help'])
    args.func(args)

    print('Done')
