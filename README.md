# Haptic-Attention-Model
Originally optimized for enabling a fully simulated KUKA robot to classify different objects solely based on gathered tactile data and the sensors pose, this stand-alone version of the **haptic attention model** (introduced in [1]) classifies 4 different objects using a pre-recorded dataset.

Here should be an image!

## Installation

**Required packages**:
1. [Tensorflow](https://www.tensorflow.org/)
2. [H5Py](https://www.h5py.org/) 
3. [Matplotlib](https://matplotlib.org/)

## Usage
The parameters for the training are all defined in the configuration files as
`run_1Glance.py`.

In order to start the training just type:
```
python run_1Glance.py 1
```
The additional argument is defining the ID of the learning run. The Results are saved within the Folder `Results_LSTM_1Glances/1/`. Thus the Folder is named after the used Memory unit (LSTM or MLP), the number of used glances. The name of the subfolder is then given by the learning runs ID. Within the folder the Model is saved (`/model/`), together with a file containing the location and position data of the sensor (`locations.hdf5`), Heatmaps of the generated location-orientation profile, a file containing the results (`results.json`) and a the data for visualizing the training information with `tensorboard` within the folder  (`/summary/`).

## Evaluation
During training information about the current losses, accuracy 
and the behavior of the location network can be gathered using `tensorboard`. 
```
tensorboard --logdir=./summary
```

To create images of the glimpses that the network uses after training, simply execute the evaluation script.
The first parameter is the name of the configuration file and the second is the path to the network model.
```
evaluate.py run_mnist ./model/
```

To plot the accuracy of the classification over the number of trained epochs use the plotting script. 
```
python plot.py ./results.json
```

## The Dataset

Stored in `objects.hdf5`, the dataset consists of the haptic data, together with the position and orientation of the myrmex sensor that were pre-recorded within the simulation.

--------
[1] Fleer, S., Moringen, A., Klatzky, R. L., & Ritter, H.  „Learning efficient haptic shape exploration with a rigid tactile sensor array.“ [arXiv preprint](https://arxiv.org/abs/1902.07501) arXiv:1902.07501. (2019).