import time
import numpy as np

from pathlib import Path

from cross_section_tree_ring_detection.cross_section_tree_ring_detection import (sampling_edges, connect_chains,
                                                                                 postprocessing, chain_2_labelme_json,
                                                                                 save_config, saving_results)
from cross_section_tree_ring_detection.chain import visualize_selected_ch_and_chains_over_image_
from filter_edges import filter_edges
from preprocessing import preprocessing
from urudendro.image import load_image, write_image

from model import deep_learning_edge_detector


def remove_duplicated_elements(l_ch_s, l_nodes_s):
    """
    This control must be done becuase the find contour method of opencv can return the same contour multiple times or with a high degree of overlapping.
    :param l_ch_s:
    :return:
    """
    l_ch_s_aux = []
    #sorted by size l_ch_s
    l_ch_s = sorted(l_ch_s, key=lambda x: len(x.l_nodes), reverse=True)
    for chain_1 in l_ch_s:
        nodos_in_common = False
        for chain_2 in l_ch_s_aux:
            if chain_1 == chain_2:
                if chain_1.id != chain_2.id:
                    #nodes in common but different id. Chain that we want to delete
                    nodos_in_common = True

                continue
            #check if they have nodes in common
            for node in chain_1.l_nodes:
                if node in chain_2.l_nodes:
                    nodos_in_common = True
                    break

        if not nodos_in_common:
            l_ch_s_aux.append(chain_1)




    if len(l_ch_s_aux) != len(l_ch_s):
        print(f"Removed {len(l_ch_s) - len(l_ch_s_aux)} duplicated chains")
        l_ch_s = l_ch_s_aux
        #chaing the chain id
        for i, chain in enumerate(l_ch_s):
            chain.change_id(i)
            chain.label_id = chain.id
        l_nodes_s = []
        for chain in l_ch_s:
            l_nodes_s += chain.l_nodes



    return l_ch_s, l_nodes_s

def control_check_duplicated_chains_in_list(l_within_chains):
    """
    Check duplicated chains in list
    @param l_within_chains: list of chains
    @return:
    """
    l_ch_s_aux = []
    for chain in l_within_chains:
        if chain not in l_ch_s_aux:
            l_ch_s_aux.append(chain)
    if len(l_ch_s_aux) != len(l_within_chains):
        raise Exception("Duplicated chains in list")

def DeepTreeRingDetection(im_in, cy, cx, sigma, th_low, th_high, height, width, alpha, nr, mc, weights_path,
                      debug= False, debug_image_input_path=None, debug_output_dir=None):
    """
    Method for delineating tree ring over pine cross sections images. Implements Algorithm 1 from the paper.
    @param im_in: segmented input image. Background must be white (255,255,255).
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param sigma: Canny edge detector gausssian kernel parameter
    @param th_low: Low threshold on the module of the gradient. Canny edge detector parameter.
    @param th_high: High threshold on the module of the gradient. Canny edge detector parameter.
    @param height: img_height of the image after the resize step
    @param width: width of the image after the resize step
    @param alpha: Edge filtering parameter. Collinearity threshold
    @param nr: rays number
    @param mc: min ch_i length
    @param debug: boolean, debug parameter
    @param debug_image_input_path: Debug parameter. Path to input image. Used to write labelme json.
    @param debug_output_dir: Debug parameter. Output directory. Debug results are saved here.
    @return:
     - l_rings: Final results. Json file with rings coordinates.
     - im_pre: Debug Output. Preprocessing image results
     - m_ch_e: Debug Output. Intermediate results. Devernay curves in matrix format
     - l_ch_f: Debug Output. Intermediate results. Filtered Devernay curves
     - l_ch_s: Debug Output. Intermediate results. Sampled devernay curves as Chain objects
     - l_ch_s: Debug Output. Intermediate results. Chain lists after connect stage.
     - l_ch_p: Debug Output. Intermediate results. Chain lists after posprocessing stage.
    """
    to = time.time()

    # Line 1 Preprocessing image. Algorithm 1 in the supplementary material. Image is  resized, converted to gray
    # scale and equalized
    im_pre, cy, cx = preprocessing(im_in, height, width, cy, cx)
    # Line 2 Edge detector module. Algorithm: A Sub-Pixel Edge Detector: an Implementation of the Canny/Devernay Algorithm,
    m_ch_e, gx, gy = deep_learning_edge_detector(im_pre,  weights_path=weights_path, output_dir=Path(debug_output_dir),
                                                  cy=cy, cx=cx, debug=debug)
    #conver im_pre to gray scale
    import cv2
    im_pre = cv2.cvtColor(im_pre, cv2.COLOR_BGR2GRAY)
    im_in = cv2.cvtColor(im_in, cv2.COLOR_RGB2BGR)
    # Line 3 Edge filtering module. Algorithm 4 in the supplementary material.
    l_ch_f = filter_edges(m_ch_e, cy, cx, gx, gy, alpha, im_pre)
    # Line 4 Sampling edges. Algorithm 6 in the supplementary material.
    l_ch_s, l_nodes_s = sampling_edges(l_ch_f, cy, cx, im_pre, mc, nr, debug=debug)
    #return im_in, im_pre, m_ch_e, l_ch_f, l_ch_s, [], [], []
    if debug:
        visualize_selected_ch_and_chains_over_image_(
            l_ch_s, [], img=im_pre, filename=f'{debug_output_dir}/chains_origin.png')
    l_ch_s, l_nodes_s = remove_duplicated_elements(l_ch_s, l_nodes_s)
    if debug:
        visualize_selected_ch_and_chains_over_image_(
            l_ch_s, [], img=im_pre, filename=f'{debug_output_dir}/chains_with_no_duplication.png')


    #control_check_duplicated_chains_in_list(l_ch_s)
    # Line 5 Connect chains. Algorithm 7 in the supplementary material. Im_pre is used for debug purposes
    debug = False
    l_ch_c,  l_nodes_c = connect_chains(l_ch_s, cy, cx, nr, debug, im_pre, debug_output_dir)
    #control_check_duplicated_chains_in_list(l_ch_c)
    # Line 6 Postprocessing chains. Algorithm 19 in the paper. Im_pre is used for debug purposes
    debug= False
    l_ch_p = postprocessing(l_ch_c, l_nodes_c, debug, debug_output_dir, im_pre)
    # Line 7
    debug_execution_time = time.time() - to
    l_rings = chain_2_labelme_json(l_ch_p, height, width, cy, cx, im_in, debug_image_input_path, debug_execution_time)

    return im_in, im_pre, m_ch_e, l_ch_f, l_ch_s, l_ch_c, l_ch_p, l_rings


def main(args):
    save_config(args, args.root, args.output_dir)
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input image {args.input} not found")

    im_in = load_image(args.input)
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    res = DeepTreeRingDetection(im_in, args.cy, args.cx, args.sigma, args.th_low, args.th_high, args.hsize, args.wsize,
                            args.edge_th, args.nr, args.min_chain_length, args.weights_path, args.debug,
                            args.input, args.output_dir)

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
    parser.add_argument("--edge_th", type=int, required=False, default=180)
    parser.add_argument("--th_high", type=int, required=False, default=20)
    parser.add_argument("--th_low", type=int, required=False, default=5)
    parser.add_argument("--min_chain_length", type=int, required=False, default=2)
    parser.add_argument("--debug", type=int, required=False)

    args = parser.parse_args()
    main(args)

