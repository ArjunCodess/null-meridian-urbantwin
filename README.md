# Null Meridian at the UrbanTwin Sim2Real LiDAR Challenge

Arjun Vijay Prakash, Independent Researcher. This repository contains the CompactPillar-v4 detector and proposal-relative ExtraTrees refiner used in the signed V2X-Real and LUMPI entries prepared for the IEEE Big Data Cup 2026.

## Reproduce the submission payload

Use Python 3.12. The reference machine used Windows, an RTX 4060, and PyTorch 2.9.1 with CUDA 12.6. Obtain the official starter kit from the Files tab of each [V2X-Real](https://www.codabench.org/competitions/16685/) or [LUMPI](https://www.codabench.org/competitions/16684/) competition. Supply your own authorized copy of `detection_test_frame_ids.json` and `detection_test_frames/`.

```sh
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python reproduce.py v2x --kit /path/to/v2x-kit --output work/v2x/predictions.json
python reproduce.py lumpi --kit /path/to/lumpi-kit --output work/lumpi/predictions.json
```

Each command checks the frozen weights and synthetic cloud hashes before running 50-frame XYZ inference. It compares the resulting prediction hash with the archived entry. You can run on CPU with `--device cpu`, although runtime and numerical results can differ. The archived entries contain 2,185 V2X-Real and 3,780 LUMPI detections. A fresh CUDA replay on the available machine produced 2,186 and 3,781 detections, respectively. The replay regenerates the two submission layouts and reproduces the method, with the retained archives providing the exact original prediction bytes.

The `assets/<track>/synthetic/` directories contain the exact 50 float32 XYZ realism clouds from the corresponding signed entry. Assemble a local submission-format ZIP with your own declaration:

```sh
python -m src.package_submission --track v2x --synthetic assets/v2x/synthetic --predictions work/v2x/predictions.json --frame-ids /path/to/v2x-kit/detection_test_frame_ids.json --declaration /path/to/signed-declaration.txt --output work/v2x/replay.zip
python -m src.package_submission --track lumpi --synthetic assets/lumpi/synthetic --predictions work/lumpi/predictions.json --frame-ids /path/to/lumpi-kit/detection_test_frame_ids.json --declaration /path/to/signed-declaration.txt --output work/lumpi/replay.zip
```

To verify a privately held original archive byte-for-byte:

```sh
python reproduce.py v2x --archive /path/to/Null_Meridian_V2X_SIGNED.zip
python reproduce.py lumpi --archive /path/to/Null_Meridian_LUMPI_SIGNED.zip
```

## Synthetic inputs and training

The synthetic inputs derive from Muhammad Shahbaz and Shaurya Agarwal's CC0 releases [UT-V2X-Real-IC](https://doi.org/10.7910/DVN/N6N4UR) and [UT-LUMPI](https://doi.org/10.7910/DVN/D9SSWD). The angular adapter uses 0.2° azimuth bins and 0.4° elevation bins. It keeps the nearest return per cell and applies zero added dropout, range noise, and vertical shift. The 64 V2X-Real and 56 LUMPI preprocessed training frames and their labels are in `assets/<track>/training/`. Their manifests use paths relative to the repository root and verify the contents. Some synthetic sequence indices can coincide with real frame numbers. The source hashes and track metadata identify them as separate simulated records.

Run these commands from the repository root for a new fit, using `v2x` or `lumpi` for TRACK. The detector selects the lowest synthetic validation loss over 64 epochs with seed 1234. The refiner uses the earlier 80% of sequence frames and must use the checkpoint from the same training manifest.

```sh
python -m src.detector train --track TRACK --manifest assets/TRACK/training.json --output work/TRACK/detector --architecture v4 --selection-metric loss --epochs 64 --seed 1234 --device cuda
python -m src.box_refiner --track TRACK --manifest assets/TRACK/training.json --detector work/TRACK/detector/best.pt --output work/TRACK/refiner --device cuda
```

The shipped weights preserve the original fit and are paired by checkpoint checksum. Loading the refiner uses joblib, so use the supplied, verified refiner files.

## Results

These are the participant-recorded Codabench metrics for Null Meridian. A ranked Final-phase submission ID was unavailable when preparing the reports. Exact server scoring requires the organizers' private references.

| Track | Combined | 3D mAP at IoU 0.5 | Realism |
| --- | ---: | ---: | ---: |
| V2X-Real | 0.3869 | 0.0378 | 0.8864 |
| LUMPI | 0.3064 | 0.0111 | 0.7420 |

Real clouds are used solely as frozen inference inputs obtained through the challenge. All training, validation, model selection, and threshold choices used synthetic data. The public package includes synthetic assets and checksums. Test clouds, test predictions, signed declarations, and organizer-only reference labels remain with their authorized holders.

