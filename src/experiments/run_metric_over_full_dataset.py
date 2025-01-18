"""
In order to run metric please visit: https://github.com/hmarichal93/uruDendro and install urudendro package
"""
from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon
from urudendro.labelme import AL_LateWood_EarlyWood, resize_annotations
from urudendro.metric_influence_area import main as metric
from urudendro.io import load_json

def get_center_pixel(annotation_path):
    al = AL_LateWood_EarlyWood(annotation_path, None)
    shapes = al.read()
    points = shapes[0].points
    pith = Polygon(points).centroid
    cx, cy = pith.coords.xy
    return cy[0], cx[0]



def main_automatic(root_database = "/data/maestria/resultados/deep_cstrd/pinus_v1/test",  results_path="/data/maestria/resultados/deep_cstrd_pinus_v1_test/inbd/inference/inbd_results/models_/inbd_urudendro_labels"):
    metadata_filename = Path(root_database) / 'dataset_ipol.csv'
    images_dir = Path(root_database) / "images/segmented"
    gt_dir = Path(root_database) / "annotations/labelme/images"
    results_path = Path(results_path)
    results_path.mkdir(exist_ok=True)

    metadata = pd.read_csv(metadata_filename)

    df = pd.DataFrame(columns=["Sample", "Precision", "Recall", "F1", "RMSE", "TP", "FP", "TN", "FN"])
    #return f"{results_path}/results.csv"
    for idx in range(metadata.shape[0]):
        row = metadata.iloc[idx]
        sample = row.Imagen

        img_path = Path(f"{images_dir}/{sample}.png")
        if not img_path.exists():
            continue
        dt = results_path / f"{sample}/labelme.json"


        dt = results_path / f"{sample}/{sample}.json"


        gt = Path(f"{gt_dir}/{sample}.json")
        cx = row.cy
        cy = row.cx
        img_resized = results_path / f"{sample}/image.png"
        dt = resize_annotations( str(img_resized), str(img_path),str(dt))
        output_sample_dir = results_path / sample
        output_sample_dir.mkdir(parents=True, exist_ok=True)
        P, R, F, RMSE, TP, FP, TN, FN = metric(str(dt), str(gt), str(img_path), str(output_sample_dir),0.6,  cy, cx)

        dt_data = load_json(str(dt))
        exec_time = None#dt_data["exec_time(s)"]

        df = pd.concat([df, pd.DataFrame([{"Sample": sample, "Precision": P, "Recall": R, "F1": F,
                                           "RMSE": RMSE, "TP": TP, "FP": FP, "TN": TN, "FN": FN, "exec_time": exec_time}])],
                       ignore_index=True)
    df.to_csv(f"{results_path}/results.csv", index=False)

    return f"{results_path}/results.csv"

def compute_statics(results_path):

    df = pd.read_csv(results_path)
    #df_stats = pd.DataFrame(columns=["Model",  "Precision", "Recall", "F1", "RMSE", "TP", "FP",  "FN"])
    stats =df[["Precision", "Recall","F1", "RMSE", "TP", "FP", "FN", "exec_time"]].mean()
    df_stats = pd.DataFrame({"P": [stats["Precision"]], "R": [stats["Recall"]],"F1": [stats["F1"]], "RMSE": [stats["RMSE"]], "TP": [stats["TP"]], "FP": [stats["FP"]],
                           "FN": [stats["FN"]], "exec_time": [stats["exec_time"]]})
    #df_stats = pd.concat([df_stats, df_aux ])

    df_stats.to_csv(Path(results_path).parent / "results_stats.csv", index=False)

if __name__ == "__main__":
    res_path = main_automatic()
    compute_statics(res_path)
