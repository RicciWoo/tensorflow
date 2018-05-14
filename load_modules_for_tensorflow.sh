module load anaconda3/4.4.0
source activate vision3
module load craype-accel-nvidia35
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/.conda_envs/vision3/lib