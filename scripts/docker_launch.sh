#!/usr/bin/env bash

# Output colors
NORMAL="\\033[0;39m"
RED="\\033[1;31m"
BLUE="\\033[1;34m"

# Names to identify images and containers of this app
IMAGE_NAME='wind_scripts'
HOMEDIR="/workspace"


log() {
  echo -e "$BLUE > $1 $NORMAL"
}

error() {
  echo ""
  echo -e "$RED >>> ERROR - $1 $NORMAL"
}


build() {
    log "Building image!"
    sudo docker build -t $IMAGE_NAME \
                    --build-arg GIT_USER=$(cat ~/.github/user) \
                    --build-arg GIT_TOKEN=$(cat ~/.github/token) \
                    .

    [ $? != 0 ] && error "Docker image build failed !" && exit 100
}

deploy() {
    log "Deploy to debug!"

    sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 -it --rm \
         -v /home/rodrigo/datadrive/wind_power/data/:/datadrive/wind_power/ \
         -v /home/rodrigo/datadrive/wind_power/results/15min/:/workspace/15min/ \
         --name RNNScript_15min_$2_$3 \
         $IMAGE_NAME \
         bash -c "python RNNScript.py --data_path /datadrive/wind_power/wind_15min.csv \
                                      --SCRIPTS_FOLDER /workspace/15min \
                                      --model $2 \
                                      --file $2_$3 \
                                      --predict_steps $3"

    [ $? != 0 ] && error "Failed!" && exit 101
}

RNNScript() {
    log "Deploy 3 RNN scrpits using $2 cell for 15 minutes interval!"

    sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -it --rm -d \
         -v /home/rodrigo/datadrive/wind_power/data/:/datadrive/wind_power/ \
         -v /home/rodrigo/datadrive/wind_power/results/15min/:/workspace/15min/ \
         --name RNNScript_15min_$2 \
         $IMAGE_NAME \
         bash -c "python RNNScript.py --data_path /datadrive/wind_power/wind_15min.csv \
                                      --SCRIPTS_FOLDER /workspace/15min \
                                      --model $2 \
                                      --file $2_Model"

    [ $? != 0 ] && error "Failed!" && exit 101
}

WavenetScript() {
    log "Deploy 3 Wavenet Script using $2 cell for 15 minutes interval!"
    for VARIABLE in 4 24 96
    do
        sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -it --rm -d \
             -v /home/rodrigo/datadrive/wind_power/data/:/datadrive/wind_power/ \
             -v /home/rodrigo/datadrive/wind_power/results/15min/:/workspace/15min/ \
             --name RNNScript_15min_$2_$VARIABLE \
             $IMAGE_NAME \
             bash -c "python WaveNetScript.py --data_path /datadrive/wind_power/wind_15min.csv \
                                          --SCRIPTS_FOLDER /workspace/15min \
                                          --file $2_$VARIABLE \
                                          --predict_steps $VARIABLE"
    done
    [ $? != 0 ] && error "Failed!" && exit 101
}

EncDecScript() {
    log "Deploy 3 Encoder-Decoder Script using $2 cell for 15 minutes interval!"
    for VARIABLE in 4 24 96
    do
        sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -it --rm -d \
             -v /home/rodrigo/datadrive/wind_power/data/:/datadrive/wind_power/ \
             -v /home/rodrigo/datadrive/wind_power/results/15min/:/workspace/15min/ \
             --name RNNScript_15min_$2_$VARIABLE \
             $IMAGE_NAME \
             bash -c "python EncDecScript.py --data_path /datadrive/wind_power/wind_15min.csv \
                                             --SCRIPTS_FOLDER /workspace/15min \
                                             --file EncDec_$2_$VARIABLE  \
                                             --predict_steps $VARIABLE"
    done
    [ $? != 0 ] && error "Failed!" && exit 101
}

EncDecAttentionScript() {
    log "Deploy 3 Encoder-Decoder Script using $2 cell and attention syste for 15 minutes interval!"
    for VARIABLE in 4 24 96
    do
        sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -it --rm -d \
             -v /home/rodrigo/datadrive/wind_power/data/:/datadrive/wind_power/ \
             -v /home/rodrigo/datadrive/wind_power/results/15min/:/workspace/15min/ \
             --name RNNScript_15min_$2_$VARIABLE \
             $IMAGE_NAME \
             bash -c "python EncDecScript.py --data_path /datadrive/wind_power/wind_15min.csv \
                                             --SCRIPTS_FOLDER /workspace/15min \
                                             --use_attention True \
                                             --file EncDecAttention_$2_$VARIABLE  \
                                             --predict_steps $VARIABLE"
    done
    [ $? != 0 ] && error "Failed!" && exit 101
}


RNNScriptJaguar() {
    log "Deploy 3 RNN scrpits using $1 cell for 15 minutes interval!"

    for VARIABLE in 4 24 96
    do
        sudo docker run --runtime=nvidia -it --rm -d \
             -v /datadrive/wind_power/data/:/datadrive/wind_power/ \
             -v /datadrive/wind_power/results/15min/:/workspace/15min/ \
             --name RNNScript_15min_$1_$VARIABLE \
             $IMAGE_NAME \
             bash -c "python RNNScript.py --data_path /datadrive/wind_power/wind_15min.csv \
                                          --SCRIPTS_FOLDER /workspace/15min \
                                          --model $1 \
                                          --file $1_$VARIABLE \
                                          --predict_steps $VARIABLE"
    done

    [ $? != 0 ] && error "Failed!" && exit 101
}

EncDecScriptJaguar() {
    log "Deploy 3 Encoder-Decoder Script using $1 cell for 15 minutes interval!"
    for VARIABLE in 4 24 96
    do
        sudo docker run --runtime=nvidia -it --rm -d \
             -v /datadrive/wind_power/data/:/datadrive/wind_power/ \
             -v /datadrive/wind_power/results/15min/:/workspace/15min/ \
             --name RNNScript_15min_$1_$VARIABLE \
             $IMAGE_NAME \
             bash -c "python EncDecScript.py --data_path /datadrive/wind_power/wind_15min.csv \
                                             --SCRIPTS_FOLDER /workspace/15min \
                                             --file EncDecAttention_$1_$VARIABLE  \
                                             --predict_steps $VARIABLE"
    done
    [ $? != 0 ] && error "Failed!" && exit 101
}

$*
