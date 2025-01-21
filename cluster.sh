#!/bin/bash
#SBATCH --job-name=deep_cstrd
#SBATCH --ntasks=8
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

# -------------------------------------------------------
#other variables
#NODE_RESULTADOS_DIR=$LOCAL_NODE_DIR/inbd/resultados
#NODE_DATASET_DIR=$LOCAL_NODE_DIR/inbd/EH

#NODE_MODEL_RESULTADOS_DIR=$NODE_RESULTADOS_DIR/model
#stdout_file="$HOME_RESULTADOS_DIR/stdout.txt"
#stderr_file="$HOME_RESULTADOS_DIR/stderr.txt"
# Define a function to check the result of a command
check_command_result() {
    # Run the command passed as an argument
    "$@"

    # Check the exit status
    if [ $? -eq 0 ]; then
        echo "Command was successful."
    else
        echo "Command failed with an error."
        exit 1
    fi
}

####Prepare directories
#rm -rf $NODE_DATASET_DIR
#rm -rf $NODE_RESULTADOS_DIR
#rm -rf $HOME_RESULTADOS_DIR

#check_command_result mkdir -p $NODE_DATASET_DIR
#check_command_result mkdir -p $NODE_RESULTADOS_DIR

####Move dataset to node local disk
#check_command_result cp  -r $HOME_DATASET_DIR $NODE_DATASET_DIR


# -------------------------------------------------------
# Run the program
cd $ROOT_DIR
python src/train.py --dataset_dir $DATASET_DIR  --logs_dir src/runs/$LOGS_DIR --model_type $MODEL_TYPE


# -------------------------------------------------------
#copy results to HOME
#mkdir -p $HOME_RESULTADOS_DIR
#cp -r $NODE_RESULTADOS_DIR/* $HOME_RESULTADOS_DIR
#cp -r $NODE_DATASET_DIR/* $HOME_RESULTADOS_DIR
#delete temporal files
#rm -rf $NODE_RESULTADOS_DIR
#rm -rf $NODE_DATASET_DIR