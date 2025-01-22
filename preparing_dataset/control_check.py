from pathlib import Path 

def check_if_directories_have_files_with_same_name(dir_1: Path, dir_2: Path):
    """check if there are files in common between two directories"""
    files_dir_1 = set([file.name for file in dir_1.iterdir()])
    print("files in dir 1: ", files_dir_1)
    files_dir_2 = set([file.name for file in dir_2.iterdir()])
    print("files in dir 2: ", files_dir_2)
    common_files = files_dir_1.intersection(files_dir_2)
    if common_files:
        print(f"Common files: {common_files}")
        return False
    else:
        print("No common files")
    return True


def check_files(root_dir = Path("/data/maestria/resultados/deep_cstrd/pinus_v1_1500")):
    dir1 = root_dir / "train/images/segmented"
    dir2 = root_dir / "val/images/segmented"
    check_if_directories_have_files_with_same_name(dir1, dir2)
    dir2 = root_dir / "test/images/segmented"
    check_if_directories_have_files_with_same_name(dir1, dir2)

    check_if_directories_have_files_with_same_name(dir1, dir1)
    print("\n\n\n")
    return

def main():
    check_files( Path("/data/maestria/resultados/deep_cstrd/pinus_v1_1500"))
    check_files(Path("/data/maestria/resultados/deep_cstrd/pinus_v2_1500"))

    return

if __name__ == "__main__":
    main()