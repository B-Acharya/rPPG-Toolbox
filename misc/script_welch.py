import subprocess
import pathlib


basePath = pathlib.Path("/homes/bacharya/rPPG-Toolbox/configs/welch_configs")
# basePath = pathlib.Path("/homes/bacharya/rPPG-Toolbox/configs/exp3_pure")
welch_configs = sorted(list(basePath.rglob("*.yaml")))

for path in welch_configs:
    fold = path.stem.split("_")[-2]
    if "CMBP" in str(path):
        print("CMBP done")
        continue
    if "COHFACE" in str(path):
        print("COHFACE done")
        continue
    if "DeepPhys" in str(path):
        print("deep done")
        continue
    if "PhysFormer" in str(path):
        print("Physformer done")
        continue
    if "Physnet" in str(path):
        print(fold)
        if int(fold) == 9:
            pass
        else:
            continue

    #
    # if "COHFACE_COHFACE_COHFACE_DeepPhys" in str(path):
    #     print("TSCAN models are missing")
    #     continue
    # if "COHFACE_COHFACE_COHFACE_Tscan" in str(path):
    #     print("cohface tscan models are missing")
    #     continue
    # if "COHFACE_COHFACE_COHFACE_PhysFormer" in str(path):
    #     print("cohface physformer models are missing")
    #     continue
    # if "COHFACE_COHFACE_COHFACE_Physnet" in str(path):
    #     print("cohface physnet models are missing")
    #     continue
    try:
        print("running on ", path)
        result = subprocess.run(["/homes/bacharya/miniconda3/envs/rppg-toolbox/bin/python3", "/homes/bacharya/rPPG-Toolbox/main.py", "--config_file", str(path), "--FOLD", fold], capture_output=True )
    except FileNotFoundError as exc:
        print("The models do not exists for this path", fold)
    if "FileNotFoundError" in str(result.stderr):
        print(result.args)
