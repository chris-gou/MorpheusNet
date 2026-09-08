#!/bin/bash

CONFIGS=("DOD-H_30_3" "DOD-H_30_2" "DOD-H_30_1" "DOD-H_30_05" "DOD-H_30_075" "DOD-H_10_3" "DOD-H_10_2" "DOD-H_10_1" "DOD-H_10_05" "DOD-H_10_075" "DOD-H_5_3" "DOD-H_5_2" "DOD-H_5_1" "DOD-H_5_05" "DOD-H_5_075")
CONFIGS_PATH="configs/ablations"
CONFIGS_PHYSIO=("PHYSIO_30_3" "PHYSIO_30_2" "PHYSIO_30_1" "PHYSIO_30_05" "PHYSIO_30_075" "PHYSIO_10_3" "PHYSIO_10_2" "PHYSIO_10_1" "PHYSIO_10_05" "PHYSIO_10_075" "PHYSIO_5_3" "PHYSIO_5_2" "PHYSIO_5_1" "PHYSIO_5_05" "PHYSIO_5_075")

# for config in "${CONFIGS[@]}"; do
#    echo "Running $CONFIGS_PATH/${config[@]}.json"
#    python trainer.py --config "configs/DOD-H.json" --override "$CONFIGS_PATH/${config[@]}.json" 
# done

# for config in "${CONFIGS_PHYSIO[@]}"; do
#    echo "Running $CONFIGS_PATH/${config[@]}.json"
#    python trainer.py --config "configs/PHYSIO.json" --override "$CONFIGS_PATH/${config[@]}.json" --resume
# done

# for config in "${CONFIGS[@]}"; do
#  echo "Running $CONFIGS_PATH/${config[@]}.json"
#    python test.py --config "configs/DOD-H.json" --override "$CONFIGS_PATH/${config[@]}.json"
# done

# for config in "${CONFIGS_SEDF78[@]}"; do
#     echo "Running $CONFIGS_PATH/${config[@]}.json"
#     python trainer.py --config "configs/SEDF78.json" --override "$CONFIGS_PATH/${config[@]}.json" 
# done

# for config in "${CONFIGS_SEDF20[@]}"; do
#     echo "Running $CONFIGS_PATH/${config[@]}.json"
#     python test.py --config "configs/SEDF20.json" --override "$CONFIGS_PATH/${config[@]}.json"
# done

# python trainer.py --config "configs/PHYSIO.json" --override "configs/ablations/PHYSIO_30_3_new_data_old_resnet.json" --name "seq_set_files" ;
# python trainer.py --config "configs/PHYSIO.json" --override "configs/ablations/PHYSIO_30_3_new_data_new_resnet.json" --name "seq_set_files" ;
# python trainer.py --config "configs/PHYSIO.json" --override "configs/ablations/PHYSIO_30_3_mnt_data_new_resnet.json" --name "seq_set_files" --resume;
# python trainer.py --config "configs/PHYSIO.json" --override "configs/ablations/PHYSIO_30_3_new_data_old_resnet.json" --name "seq_set_files" --resume;
# python trainer.py --config "configs/PHYSIO.json" --override "configs/ablations/PHYSIO_30_3_harm_data_old_resnet.json" --name "seq_set_files" ;

python trainer.py --config "configs/PHYSIO.json" --override "configs/ablations/PHYSIO_30_3_new_data_new_resnet.json" --name "no_bandpass" --resume;
python trainer.py --config "configs/PHYSIO.json" --override "configs/ablations/PHYSIO_30_3_new_data_old_resnet.json" --name "no_bandpass" ;
python trainer.py --config "configs/PHYSIO.json" --override "configs/ablations/PHYSIO_30_3_mor_data_new_resnet.json" --name "no_bandpass" ;
python trainer.py --config "configs/PHYSIO.json" --override "configs/ablations/PHYSIO_30_3_mor_data_old_resnet.json" --name "no_bandpass" ;

