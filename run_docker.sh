#!/bin/bash

docker rm -fv ignacio_event_reconst_pix2pix

docker run -it --gpus '"device=6,7"' --name ignacio_event_reconst_pix2pix -v /home/ignacio.bugueno/cachefs/event_reconst_pix2pix/input:/app/input -v /home/ignacio.bugueno/cachefs/event_reconst_pix2pix/output:/app/output ignacio_event_reconst_pix2pix
