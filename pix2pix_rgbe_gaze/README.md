# Pix2Pix RGBE-Gaze

This module trains a grayscale Pix2Pix model from accumulated-event images to
synchronized grayscale face targets. It is isolated from `pix2pix_unet_faces/`
and does not require Docker or volume changes.

## Container paths

The existing `run_docker.sh` mounts:

```text
Host input  -> /app/input
Host output -> /app/output
```

### Expose the existing dataset without copying it

A regular symlink is not sufficient because its absolute host destination is
not visible inside the container. Create a read-only nested bind mount on the
host before starting or recreating the container:

```bash
mkdir -p /home/ignacio.bugueno/cachefs/event_reconst_pix2pix/input/rgbe-gaze

sudo mount --bind \
  /home/ignacio.bugueno/cachefs/datasets/processed_data/reconstruction/rgbe-gaze \
  /home/ignacio.bugueno/cachefs/event_reconst_pix2pix/input/rgbe-gaze

sudo mount -o remount,bind,ro \
  /home/ignacio.bugueno/cachefs/datasets/processed_data/reconstruction/rgbe-gaze \
  /home/ignacio.bugueno/cachefs/event_reconst_pix2pix/input/rgbe-gaze

findmnt /home/ignacio.bugueno/cachefs/event_reconst_pix2pix/input/rgbe-gaze
```

The nested mount must exist before the container starts because Docker bind
mount propagation is private by default. Recreate the container with the
unchanged `run_docker.sh`; the dataset will then appear at
`/app/input/rgbe-gaze`. To remove the host-side mapping later:

```bash
sudo umount /home/ignacio.bugueno/cachefs/event_reconst_pix2pix/input/rgbe-gaze
```

Place or expose RGBE-Gaze with this structure:

```text
/app/input/rgbe-gaze/
├── event_accumulate_frames/user_1/exp1/*.png
├── event_accumulate_frames/user_1/exp2/*.png
├── ...
└── gray_frames/user_1/exp1/*.png
    ...
```

Input and target files are paired by identical paths relative to their two
root directories. Unpaired files are reported and skipped by default.

Validate paths, pairs, splits, and image loading before training:

```bash
cd /app/pix2pix_rgbe_gaze
python check_dataset.py --device dgx-1 --rep acc_events
```

Add `--strict-pairs` when any unpaired image should make validation fail.

The configuration defaults to `user_1`. Select it explicitly with:

```bash
python check_dataset.py \
  --device dgx-1 \
  --rep acc_events \
  --users user_1
```

Include every available user with:

```bash
python check_dataset.py \
  --device dgx-1 \
  --rep acc_events \
  --all-users
```

## Split used for `user_1`

```text
Train: exp1, exp2, exp3, exp4
Validation: exp5
Test: exp6
```

## Train accumulated events

Inside the running container:

```bash
cd /app/pix2pix_rgbe_gaze

python train.py \
  --device dgx-1 \
  --rep acc_events \
  --gpu 0 \
  --users user_1
```

Use both GPUs exposed by the existing container:

```bash
python train.py \
  --device dgx-1 \
  --rep acc_events \
  --gpu 0,1 \
  --users user_1
```

To train the same experiment split with every available user, replace
`--users user_1` with `--all-users`. In all-user mode, `exp1` through `exp4`
remain training, `exp5` validation, and `exp6` test for every identity.

Each fresh run receives a timestamped directory under:

```text
/app/output/pix2pix_rgbe_gaze/
```

Provide `--run-name NAME` for a stable name. The command refuses to start a new
run when that directory already exists, preventing accidental overwrites.

Resume an interrupted run with the same name:

```bash
python train.py \
  --device dgx-1 \
  --rep acc_events \
  --gpu 0,1 \
  --users user_1 \
  --run-name NAME \
  --resume /app/output/pix2pix_rgbe_gaze/NAME/checkpoints/last.pt
```

## Evaluate exp6

```bash
python evaluate.py \
  --device dgx-1 \
  --rep acc_events \
  --gpu 0 \
  --users user_1 \
  --checkpoint /app/output/pix2pix_rgbe_gaze/NAME/checkpoints/best.pt \
  --output-dir /app/output/pix2pix_rgbe_gaze/NAME/test \
  --split test
```

Evaluation writes event/generated/target images, horizontal comparisons,
per-image metrics, and a JSON summary for MSE, SSIM, and PSNR. It refuses to
write into a non-empty evaluation directory unless `--overwrite` is provided.

## Generate labeled horizontal examples

Generate eight `Input (event) | Generated | Target` comparisons from `exp6`:

```bash
python sample.py \
  --device dgx-1 \
  --rep acc_events \
  --gpu 7 \
  --users user_1 \
  --checkpoint /app/output/pix2pix_rgbe_gaze/NAME/checkpoints/best.pt \
  --split test \
  --limit 8
```

By default, images are written below `NAME/samples/test-best/`. Use
`--output-dir` to select another location. The command refuses to write into a
non-empty directory unless `--overwrite` is supplied. Use `--offset` to inspect
a later portion of the split.

## Updating the already running container

No Dockerfile or `run_docker.sh` changes are required. After committing and
pushing this module, update the repository already copied into the container:

```bash
cd /app
git pull origin main
```

Then enter `/app/pix2pix_rgbe_gaze` and run the dataset check and training
commands above.
