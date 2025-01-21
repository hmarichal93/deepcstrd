#!/bin/bash
#SBATCH --job-name=deep_cstrd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --tmp=100G
#SBATCH --mail-user=henry.marichal@fing.edu.uy

# de acuerdo a lo que quiera ejecutar puede elegir entre las siguientes tres líneas.
#SBATCH --gres=gpu:1 # se solicita una gpu cualquiera( va a tomar la primera que quede disponible indistintamente si es una p100 o una a100)


#SBATCH --partition=besteffort
#SBATCH --qos=besteffort


source /etc/profile.d/modules.sh
source /clusteruy/home/henry.marichal/miniconda3/etc/profile.d/conda.sh

# -------------------------------------------------------
#disco local SSD local al nodo. /clusteruy/home/henry.marichal se accede via NFS (puede ser realmente lento)
#el espacio local a utilizar se reserva dcon --tmp=XXXGb
ROOT_DIR=$1
DATASET_DIR=$2
LOGS_DIR=$3
MODEL_TYPE=$4
#install conda environemtnt
cd $ROOT_DIR
#conda env create -f src/environment.yml
conda activate deep_cstrd
#pip install -r src/requirements.txt

##
NODE_SSD_DIR=/scratch/henry.marichal
#copy dataset to local disk
NODE_DATASET_DIR=$NODE_SSD_DIR/dataset
NODE_LOGS_DIR=$NODE_SSD_DIR/logs
#copy dataset to local disk
rm -rf $NODE_DATASET_DIR
rm -rf $NODE_LOGS_DIR
mkdir -p $NODE_DATASET_DIR
mkdir -p $NODE_LOGS_DIR

cp -r $DATASET_DIR $NODE_DATASET_DIR

# -------------------------------------------------------
####Prepare directories


#check_command_result mkdir -p $NODE_DATASET_DIR
#check_command_result mkdir -p $NODE_RESULTADOS_DIR

####Move dataset to node local disk
#check_command_result cp  -r $HOME_DATASET_DIR $NODE_DATASET_DIR


# -------------------------------------------------------
# Run the program
cd $ROOT_DIR
python src/train.py --dataset_dir $NODE_DATASET_DIR  --logs_dir src/runs/$LOGS_DIR --model_type $MODEL_TYPE


# -------------------------------------------------------
