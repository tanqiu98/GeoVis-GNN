## Environment Setup

First please create an appropriate environment using conda: 

> conda env create -f environment.yml

> conda activate mphoi

## Download Data

Please download the necessary data from the link below, and put the 
downloaded data folder in this current directory (i.e. `./data/...`).

Link: [data](https://drive.google.com/drive/folders/1yfwItIoQrAnbnk5GTjbbfN8Ls8Ybl_hr?usp=sharing).

## Train the Model

To train the model from scratch, edit the `./conf/config.yaml` file, and depending on the selected dataset and model, also 
edit the associated model .yaml file in `./conf/models/` and the associated dataset .yaml file in `./conf/data/`. After 
editing the files, just run `python train.py`.

## Test the Model

Examples on MPHOI-72: when you get pre-trained models for all subject groups, you can get the cross-validation result by `python -W ignore predict.py --pretrained_model_dir ./outputs/mphoi72/GeoVisGNN/hs512_e40_bs16_lr0.0001_0.1_Subject45 --cross_validate`.


