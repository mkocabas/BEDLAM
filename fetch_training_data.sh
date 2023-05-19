#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# BEDLAM checkpoints
echo -e "\nYou need to register at https://bedlam.is.tue.mpg.de/"
read -p "Username (BEDLAM):" username
read -p "Password (BEDLAM):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/training_images
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps.tar' -O './data/training_images/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221010_3_1000_batch01hand_6fps.tar' -O './data/training_images/20221010_3_1000_batch01hand_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221011_1_250_batch01hand_closeup_suburb_a_6fps.tar' -O './data/training_images/20221011_1_250_batch01hand_closeup_suburb_a_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221011_1_250_batch01hand_closeup_suburb_b_6fps.tar' -O './data/training_images/20221011_1_250_batch01hand_closeup_suburb_b_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221011_1_250_batch01hand_closeup_suburb_c_6fps.tar' -O './data/training_images/20221011_1_250_batch01hand_closeup_suburb_c_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221011_1_250_batch01hand_closeup_suburb_d_6fps.tar' -O './data/training_images/20221011_1_250_batch01hand_closeup_suburb_d_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps.tar' -O './data/training_images/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps.tar' -O './data/training_images/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps.tar' -O './data/training_images/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221013_3_250_batch01hand_orbit_bigOffice_6fps.tar' -O './data/training_images/20221013_3_250_batch01hand_orbit_bigOffice_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221013_3_250_batch01hand_static_bigOffice_6fps.tar' -O './data/training_images/20221013_3_250_batch01hand_static_bigOffice_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps.tar' -O './data/training_images/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps.tar' -O './data/training_images/20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps.tar' -O './data/training_images/20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps.tar' -O './data/training_images/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221017_3_1000_batch01hand_6fps.tar' -O './data/training_images/20221017_3_1000_batch01hand_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps.tar' -O './data/training_images/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps.tar' -O './data/training_images/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221019_1_250_highbmihand_closeup_suburb_b_6fps.tar' -O './data/training_images/20221019_1_250_highbmihand_closeup_suburb_b_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221019_1_250_highbmihand_closeup_suburb_c_6fps.tar' -O './data/training_images/20221019_1_250_highbmihand_closeup_suburb_c_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221019_3-8_1000_highbmihand_static_suburb_d_6fps.tar' -O './data/training_images/20221019_3-8_1000_highbmihand_static_suburb_d_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221019_3_250_highbmihand_6fps.tar' -O './data/training_images/20221019_3_250_highbmihand_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps.tar' -O './data/training_images/20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221022_3_250_batch01handhair_static_bigOffice_30fps.tar' -O './data/training_images/20221022_3_250_batch01handhair_static_bigOffice_30fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221024_10_100_batch01handhair_zoom_suburb_d_30fps.tar' -O './data/training_images/20221024_10_100_batch01handhair_zoom_suburb_d_30fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps.tar' -O './data/training_images/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/agora_images.tar' -O './data/training_images/agora_images.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/20221019_3-8_250_highbmihand_orbit_stadium_6fps.tar' -O './data/training_images/20221019_3-8_250_highbmihand_orbit_stadium_6fps.tar' --no-check-certificate --continue


mkdir -p data/training_labels
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_labels/all_npz_12_training.zip' -O './data/training_labels/all_npz_12_training.zip' --no-check-certificate --continue
unzip data/training_labels/all_npz_12_training.zip -d data/bedlam_labels
# For 3DPW validation
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=eval_data_parsed.zip' -O './data/eval_data_parsed.zip' --no-check-certificate --continue
unzip data/eval_data_parsed.zip -d data

# For 3DPW finetuning
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=3dpw_train_smplx.npz' -O './data/training_labels/3dpw_train_smplx.npz' --no-check-certificate --continue
