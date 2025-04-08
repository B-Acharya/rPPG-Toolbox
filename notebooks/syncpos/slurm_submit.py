from simple_slurm import Slurm
import datetime


lrs = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.5, 0.05, 0.005]

# config_file_path = "/homes/bacharya/rPPG-Toolbox/configs/train_configs/PURE_PURE_PURE_DEEPPHYS_FINGER_PPG.yaml"
config_file_path = "/homes/bacharya/rPPG-Toolbox/configs/train_configs/PURE_PURE_PURE_DEEPPHYS_FACE_PPG.yaml"


for lr in lrs:

    if "FINGER" in config_file_path:
        job_name = f"FPPG-{lr}"
        output_file = f"mlflow-pure-fppg-{lr}-output.txt",
        error_file =f"mlflow-pure-fppg-{lr}-error.txt",
    elif "FACE" in config_file_path:
        job_name = f"CPPG-{lr}"
        output_file = f"mlflow-pure-cppg-{lr}-output.txt",
        error_file =f"mlflow-pure-cppg-{lr}-error.txt",
    else:
        NotImplementedError

    slurm = Slurm(
        mail_user="bacharya@techfak.uni-bielefeld.de",
        mail_type=["BEGIN", "END"],
        output=output_file,
        error=error_file,
        time=datetime.timedelta(days=1, hours=24),
        gpus=1,
        cpus_per_task=30,
        job_name=job_name
    )

    # Activate conda environment and set PYTHONPATH
    slurm.add_cmd("source /homes/bacharya/miniconda3/bin/activate")
    slurm.add_cmd("conda activate toolbox")
    slurm.add_cmd("export PYTHONPATH=${PYTHONPATH}:/homes/bacharya/rPPG-Toolbox/")

    # Print debugging information
    slurm.add_cmd("echo \"Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID using gpu id +++$CUDA_DEVICE (we may use gpu: $CUDA_VISIBLE_DEVICES on $(hostname))\"")
    slurm.add_cmd("echo \"computing on $(nvidia-smi --query-gpu=gpu_name --format=csv -i $CUDA_DEVICE | tail -n 1)\"")

    # Run the Python script
    slurm.sbatch(f"python main_syncpos.py --config_file {config_file_path} --lr {lr}")