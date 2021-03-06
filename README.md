# Haptic-Attention-Model
Originally optimized for enabling a fully simulated KUKA robot to classify different objects solely based on gathered tactile data and the sensors pose, this stand-alone version of the **haptic attention model** (introduced in [1]) classifies 4 different objects using a pre-recorded dataset.

![Objects](./images/objects.png)

## Installation

**Required packages**:
1. [Tensorflow](https://www.tensorflow.org/)
2. [H5Py](https://www.h5py.org/)
3. [Matplotlib](https://matplotlib.org/)

## Usage
The parameters for the training are all defined in the configuration files as
`run.py`.

In order to start the training just type:
```
python run.py 1 6
```
The first argument is defining the ID of the learning run and the second one the number of used glances.
For a list of all possible arguments just type `python run.py -h`. The program does not rely on any specific hardware specifications and can be executed on a regular computer.

The Results are saved within the Folder `Results_LSTM_1Glances/1/`. The Folder is named after the used Memory unit (LSTM or MLP), the number of used glances. The name of the subfolder is then given by the learning runs ID. Within the folder the Model is saved (`/model/`), together with a file containing the location and position data of the sensor (`locations.hdf5`), Heatmaps of the generated location-orientation profile, a file containing the results (`results.json`) and a the data for visualizing the training information with `tensorboard` within the folder  (`/summary/`).

## Evaluation
During training information about the current losses, accuracy
and the behavior of the location network can be gathered using `tensorboard`.
```
tensorboard --logdir=./summary
```

To plot the accuracy of the classification over the number of trained epochs use the plotting script followed by the generated .json file.
```
python plot.py ./results.json
```
When starting the training using the configuration `python run.py 1 10` (i.e. training on 10 glances), the result should be similar to
![Objects](./images/plot.png)

## The Dataset

Stored in `objects.hdf5`, the dataset contains the haptic data, together with the position and orientation of the myrmex sensor that were pre-recorded within the simulation.

--------
[1] Fleer S, Moringen A, Klatzky RL, Ritter H (2020) **Learning efficient haptic shape exploration with a rigid tactile sensor array**. PLOS ONE 15(1): e0226880. https://doi.org/10.1371/journal.pone.0226880
