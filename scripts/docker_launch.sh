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
  echo -e "$RED >>> ERROR - $1$NORMAL"
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

    for VARIABLE in 4 24 96
    do
        sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -it --rm -d \
             -v /home/rodrigo/datadrive/wind_power/data/:/datadrive/wind_power/ \
             -v /home/rodrigo/datadrive/wind_power/results/15min/:/workspace/15min/ \
             --name RNNScript_15min_$2_$VARIABLE \
             $IMAGE_NAME \
             bash -c "python RNNScript.py --data_path /datadrive/wind_power/wind_15min.csv \
                                          --SCRIPTS_FOLDER /workspace/15min \
                                          --model $2 \
                                          --file $2_$VARIABLE \
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

bash() {
  log "BASH"
  docker run -it --rm -v $(pwd):/app $IMAGE_NAME /bin/bash
}

stop() {
  docker stop $CONTAINER_NAME
}

start() {
  docker start $CONTAINER_NAME
}

remove() {
  log "Removing previous container $CONTAINER_NAME" && \
      docker rm -f $CONTAINER_NAME &> /dev/null || true
}

help() {
  echo "-----------------------------------------------------------------------"
  echo "                      Available commands                              -"
  echo "-----------------------------------------------------------------------"
  echo -e -n "$BLUE"
  echo "   > build - To build the Docker image"
  echo "   > npm - To install NPM modules/deps"
  echo "   > bower - To install Bower/Js deps"
  echo "   > jkbuild - To build Jekyll project"
  echo "   > grunt - To run grunt task"
  echo "   > jkserve - To serve the project/blog on 127.0.0.1:4000"
  echo "   > install - To execute full install at once"
  echo "   > stop - To stop main jekyll container"
  echo "   > start - To start main jekyll container"
  echo "   > bash - Log you into container"
  echo "   > remove - Remove main jekyll container"
  echo "   > help - Display this help"
  echo -e -n "$NORMAL"
  echo "-----------------------------------------------------------------------"

}

$*
