#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --partition=g100_usr_prod
#SBATCH --account=uBS23_InfGer
#SBATCH --time=00:07:00
#SBATCH --mem=64G
#SBATCH --error=ac2Stable.err
#SBATCH --output=a2cStable.out
## SBATCH --ntasks-per-socket=1
##SBATCH --ntasks-per-node=1

#source /g100_work/PROJECTS/spack/v0.16/install/0.16.2/linux-centos8-skylake_avx512/gcc-8.3.1/anaconda3-2020.07-l2bohj4adsd6r2oweeytdzrgqmjl64lt/etc/profile.d/conda.sh

#conda activate a2cBaseline

python A2CStable.py