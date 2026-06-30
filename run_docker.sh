#!/bin/bash

docker rm -fv ignacio_event_reconst_pix2pix

docker run -it \
  --gpus '"device=0,1,2,3,4,5,6,7"' \
  --name ignacio_event_reconst_pix2pix \
  --shm-size=32g \
  --ipc=host \
  -v /home/ignacio.bugueno/cachefs/event_reconst_pix2pix/input:/app/input \
  -v /home/ignacio.bugueno/cachefs/event_reconst_pix2pix/output:/app/output \
  -v /home/ignacio.bugueno/cachefs/datasets/processed_data/reconstruction/rgbe-gaze:/app/input/rgbe-gaze:ro \
  ignacio_event_reconst_pix2pix
