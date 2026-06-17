# MorpheusNet

Repository to re-create metrics and figures for the paper: https://arxiv.org/abs/2401.10284

![MorpheusNet architecture](https://github.com/ali77sina/MorpheusNet/assets/54308350/0a8bf16f-c019-4312-a569-840654405ada)

Fig. 1: MorpheusNet is a resource-efficient deep neural net for sleep stage classification, developed to be deployed on commercially available microcontrollers.

The folders contain scripts for each dataset used in the study, together with the open-source hardware for the sleep headband.

- `DOD-H/`: Dreem Open Dataset, Healthy sub-group. 25 patients classified with single-channel EEG.
- `PCB/`: PCB manufacturing files (Gerbers, BOM, pick-and-place) for the ADS1299/Teensy acquisition board.
- `headband design/`: Fusion 360 CAD for the headband enclosure and electrode mounts.

**TODO**: add Sleep-EDF and PhysioNet 2018 challenge related scripts.
