import os
import pandas as pd
import glob

from pathlib import Path
from fpdf import FPDF
from natsort import natsorted
from tqdm import tqdm


def generate_pdf(path):
    pdf = FPDF()
    pdf.set_font('Arial', 'B', 16)

    resultado_final = "output.png"
    figures = glob.glob(f"{path}/**/{resultado_final}", recursive=True)
    idx = 0
    for fig in tqdm(natsorted(figures)):
        idx += 1
        splitted = fig.split("/")
        disco = splitted[-2]
        disco = disco.replace("_", "")

        fig_path = ""
        for sub_path in splitted[:-1]:
            fig_path += sub_path + "/"

        # fig2 = fig_path + "preprocessing_output.png"
        fig3 = fig_path + "segmentation.png"
        #fig4 = fig_path + "connect.png"
        #fig5 = fig_path + "postprocessing.png"
        fig6 = fig_path + "output.png"
        x, y = 0, 40
        height = 180

        for fig in [fig3, fig6]:
            pdf.add_page()
            pdf.cell(0, 0, disco)
            pdf.image(fig, x, y, h=height)

        #pdf.add_page()

    pdf.output(f"{path}/summary_ipol.pdf", 'F')

class TRD:
    CSTRD=1
    INBD= 2
    DEEPCSTRD=3

def main(root_database = "/data/maestria/resultados/deep_cstrd/pinus_v1/test",  results_path="/data/maestria/resultados/deep_cstrd_pinus_v1_test/deep_cstrd",
         weights_path="/home/henry/Documents/repo/fing/cores_tree_ring_detection/src/runs/pinus_v1_40_train_12_val/epoch_20/latest_model.pth",
         method=TRD.CSTRD):

    metadata_filename = Path(root_database).parent / 'dataset_ipol.csv'
    images_dir = Path(root_database) / "images/segmented"
    results_path = Path(results_path)
    results_path.mkdir(exist_ok=True, parents=True)

    metadata = pd.read_csv(metadata_filename)
    for idx in range(metadata.shape[0]):
        row = metadata.iloc[idx]
        name = row.Imagen

        img_filename = images_dir / f"{name}.png"
        if not img_filename.exists():
            img_filename = images_dir / f"{name}.jpg"
            if not img_filename.exists():
                continue
        img_res_dir = (results_path / name)
        img_res_dir.mkdir(exist_ok=True)

        cy = int(row.cy)
        cx = int(row.cx)
        #sigma = row.sigma
        sigma = 3
        if (img_res_dir / "labelme.json").exists():
            continue




        if method == TRD.CSTRD:
            print("CSTRD")
            from cross_section_tree_ring_detection.cross_section_tree_ring_detection import TreeRingDetection
            from cross_section_tree_ring_detection.io import load_image
            from cross_section_tree_ring_detection.utils import save_config, saving_results

            args = dict(cy=cy, cx=cx, sigma=3, th_low=5, th_high=20,
                        height=0, width=0, alpha=30, nr=360,
                        mc=2)

            im_in = load_image(str(img_filename))
            res = TreeRingDetection(im_in, **args)
            saving_results(res, img_res_dir, 1)

        elif method == TRD.DEEPCSTRD:
            print("DeepCSTRD")
            command = f"python main.py --input {img_filename} --sigma {sigma} --cy {cy} --cx {cx}  --root ./ --output_dir" \
                      f" {img_res_dir}  --weights_path {weights_path}"


            print(command)
            os.system(command)

        else:
            print("Method not implemented")

    generate_pdf(results_path)



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a U-Net model for image segmentation')
    parser.add_argument('--dataset_dir', type=str, default="/data/maestria/resultados/deep_cstrd/pinus_v1/test",
                        help='Path to the dataset directory')
    parser.add_argument('--results_path', type=str, default="/data/maestria/resultados/deep_cstrd_pinus_v1_test/deep_cstrd",
                        help='Path to the results directory')
    parser.add_argument('--weights_path', type=str, default="/home/henry/Documents/repo/fing/cores_tree_ring_detection/src/runs/pinus_v1_40_train_12_val/epoch_20/latest_model.pth",
                        help='Path to the weights directory')
    parser.add_argument('--method', type=int, default=TRD.CSTRD,
                        help='Method to use for tree ring detection. 1: CSTRD, 2: INBD, 3: DEEPCSTRD')
    args = parser.parse_args()

    main(args.dataset_dir, args.results_path, args.weights_path, args.method)



