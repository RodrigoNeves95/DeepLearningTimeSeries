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

RNNTrainer() {
  log "RNN run for 15 minutes interval!"

  sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -it --rm \
         -v /home/rodrigo/datadrive/wind_power/data/:/datadrive/wind_power/ \
         -v /home/rodrigo/datadrive/wind_power/results/15min/:/workspace/15min/ \
         $IMAGE_NAME \
         bash -c "python RNNScript.py --data_path /datadrive/wind_power/wind_15min.csv \
                                      --SCRIPTS_FOLDER /workspace/15min \
                                      --file RNN_$2 \
                                      --predict_steps $2"

  [ $? != 0 ] && error "Failed!" && exit 101
}

bower() {
  log "Bower install"
  docker run -it --rm -v $(pwd):/app -v /var/tmp/bower:$HOMEDIR/.bower $IMAGE_NAME \
    /bin/bash -ci "$EXECUTE_AS bower install \
      --config.interactive=false \
      --config.storage.cache=$HOMEDIR/.bower/cache"

  [ $? != 0 ] && error "Bower install failed !" && exit 102
}

jkbuild() {
  log "Jekyll build"
  docker run -it --rm -v $(pwd):/app $IMAGE_NAME \
    /bin/bash -ci "$EXECUTE_AS jekyll build"

  [ $? != 0 ] && error "Jekyll build failed !" && exit 103
}

grunt() {
  log "Grunt build"
  docker run -it --rm -v $(pwd):/app $IMAGE_NAME \
    /bin/bash -ci "$EXECUTE_AS grunt"

  [ $? != 0 ] && error "Grunt build failed !" && exit 104
}

jkserve() {
  log "Jekyll serve"
  docker run -it -d --name="$CONTAINER_NAME" -p 4000:4000 -v $(pwd):/app $IMAGE_NAME \
    /bin/bash -ci "jekyll serve -H 0.0.0.0"

  [ $? != 0 ] && error "Jekyll serve failed !" && exit 105
}

install() {
  echo "Installing full application at once"
  remove
  npm
  bower
  jkbuild
  grunt
  jkserve
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
