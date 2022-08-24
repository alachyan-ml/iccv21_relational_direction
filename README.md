# Human Object Interaction Detection in Egocentric Datasets using Spatial Relation Model

This project builds on Interaction Compass: Multi-Label Zero-Shot Learning of Human-Object Interactions via Spatial Relations

## Overview
> In this project, we evaluate the position of the HOI detection model from previous research on an egocentric video dataset Epic Kitchens.
![Image](https://github.com/hbdat/iccv21_relational_direction/raw/main/fig/schemantic_figure.png)

---
## Prerequisites
To install all the dependency packages, please run:
```
pip install -r requirements.txt
```

---
## Data Preparation
1) Please download and extract the `epic_images.tar.gz` archive into the `data/epic_kitchens`

2) Create the `features_pad` directory in the data directory for the feature extraction script to extract to. 
```
mkdir data/epic_kitchens/features_pad
```

3) Please run feature extraction scripts in `./extract_feature` folder to extract features from the last convolution layers of ResNet as region features for the attention mechanism:
```
python ./extract_feature/epic_kitchens/epic_extract_feature_map_ResNet_152_padding.py [type] --size [size] --split [split] --part [part]    
```
Where 
- `type` is one of [train, validation].
- `size` is the number of frames to skip per annotation frame grabbed.
- `split` is the number of pieces to split the full feature set into. This allows for running concurrent feature extraction on the extremely large dataset. 
- `part` is the part of the splits to extract features from. Number must be between [0, split-1].

as well as word embedding for zero-shot learning:
```
python extract_feature/epic_kitchens/epic_extract_action_object_w2v.py
```

---
## Experiments
1) To train cross attention on the extracted epic kitchens features please run the following commands:
```
# Epic Kitchens
python ./experiments/epic_kitchens/epic_kitchens_pad_CrossAttention.py --idx_GPU [gpu] --save_folder [save_folder] --lr [learning_rate] --load_model [model_path]
```

Where:
- `gpu` is the gpu number that the model will train on
- `save_folder` is the location under `./results` for the results to be placed. Suggested value is `epic_kitchens_results`.
- `lr` is the learning rate. any number under 0.001 is usually suggested. 
- `model_path` is the path to the model checkpoint that can be used by the training loop. 

There are other options in the source code, but those are left default. 

## Data Analysis. 
The `Model_Analysys.ipynb` folder contains some analysis regarding the latest experiment. 

To utilize this notebook, change the `experiment_dir` value in the first cell to the path to the `save_folder` location from the experiment that was recently run. 
If evaluating the last trained model, set `chosen_experiment` in cell 2 to `newest_experiment`. If evaluating a previous experiment, change `n` in the first cell to the nth to last experiment and set `chosen experiment` to `nth_experiment`.

This model will plot the loss graphs for the model, print the last mAP during training, and then run a validation loop on a randomly selected subset of the data. 
Then the notebook will generate a visualization that shows images that were correctly and incorrectly predicted along with the action and object attention map. 

This project builds on the work done by Dat Huynh and Ehsan Elhamifar. The citation for the paper is shown below. 
```
@article{Huynh:ICCV21,
  author = {D.~Huynh and E.~Elhamifar},
  title = {Interaction Compass: Multi-Label Zero-Shot Learning of Human-Object Interactions via Spatial Relations},
  journal = {International Conference on Computer Vision},
  year = {2021}}
```

