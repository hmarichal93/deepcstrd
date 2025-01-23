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
BATCH_SIZE=$5
TILE_SIZE=$6
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
LOCAL_LOGS_DIR="$ROOT_DIR/src/runs/$LOGS_DIR"
#copy dataset to local disk
echo "Removing old dataset from local disk"

rm -rfv $NODE_DATASET_DIR
rm -rfv $NODE_LOGS_DIR
rm -rfv $LOCAL_LOGS_DIR

echo "Copying dataset to local disk"

mkdir -p $NODE_DATASET_DIR
mkdir -p $NODE_LOGS_DIR
mkdir -p $LOCAL_LOGS_DIR

cp -rv $DATASET_DIR/* $NODE_DATASET_DIR

ls -l $NODE_DATASET_DIR


# -------------------------------------------------------
# Run the program
echo "Running the program"
cd $ROOT_DIR
python src/train.py --dataset_dir $NODE_DATASET_DIR  --logs_dir $NODE_LOGS_DIR --model_type $MODEL_TYPE --batch_size $BATCH_SIZE --tile_size $TILE_SIZE


# -------------------------------------------------------
echo "Copying logs to NFS"
cp -r $NODE_LOGS_DIR $LOCAL_LOGS_DIR

