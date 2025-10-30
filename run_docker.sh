#!/bin/bash

docker rm -fv ignacio_event_reconst_pix2pix

docker run -it \
  --gpus '"device=0,1"' \
  --name ignacio_event_reconst_pix2pix \
  --shm-size=32g \
  --ipc=host \
  -v /home/ignacio.bugueno/cachefs/event_reconst_pix2pix/input:/app/input \
  -v /home/ignacio.bugueno/cachefs/event_reconst_pix2pix/output:/app/output \
  ignacio_event_reconst_pix2pix
