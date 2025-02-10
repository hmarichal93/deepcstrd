from urudendro.labelme import AL_LateWood_EarlyWood

from pathlib import Path

def count_rings_in_dataset(dataset_dir, debug=False):

    annotations_dir = Path(dataset_dir) / "annotations"
    ring_conter = 0
    for ann in annotations_dir.rglob("*.json"):
        al = AL_LateWood_EarlyWood(ann, None)
        shapes = al.read()
        ring_conter += len(shapes)

    print(ring_conter)
    return

if __name__ == "__main__":
    count_rings_in_dataset("/data/maestria/resultados/deep_cstrd_datasets/gleditsia")