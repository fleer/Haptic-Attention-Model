"""
Author: Sascha Fleer
"""
import sys
import argparse

class PARAMETERS:
    """
    Class for specifying the parameters for
    the learning algorithm
    """

    #   Reward for correctly Identifying the object:
    REWARD = +1.
    # Number of Swipes
    GLANCES = 1
    # Number of objects
    OBJECT_NUM = 4
    #   =========================
    #   General parameters for the
    #   experiment
    #   =========================


    #   Number of learning epochs
    MAX_STEPS = 5000
    #   Batch size
    BATCH_SIZE = 64
    #   Early stopping
    EARLY_STOPPING = False
    #   Number of Performance Evaluations observing the worsening of
    #   Validation set, before stopping
    PATIENCE = 10

    #   =========================
    #   Random Location Policy
    #   =========================
    RANDOM_LOCS = False

    #   =========================
    #   Save and Load the Model Weights
    #   =========================
    LOAD_MODEL = False

    #   =========================
    #   Save the used Location and Orientation
    #   of the Myrmex Sensor
    #   =========================
    SAVE_SENSOR_POLICY = False
    HEATMAPS = False

    #   =========================
    #   Train the model or Visualize results within Gazebo
    #   Only one of the options can be active at the same time
    #   =========================
    VISUALIZATION = False
    TRAIN_GAZEBO = False

    #   =========================
    #   Network Parameters
    #   --> Number of Neurons
    #   =========================

    # Core Network: LSTM, MLP
    CORE_NET = 'LSTM'
    HIDDEN_STATE = 256
    HAPTIC_NET = 64
    LOCATION_WEIGHT = 0.4

    #   =========================
    #   Algorithm specific parameters
    #   =========================

    #   To be used optimizer:
    #   rmsprop
    #   adam
    #   adadelta
    #   sgd
    OPTIMIZER = 'sgd'
    # Momentum
    MOMENTUM = 0.9
    # Learning rate alpha
    LEARNING_RATE = 0.0008
    #LEARNING_RATE = 0.001
    # Decay type for learning rate
    #   - static
    #   - linear
    #   - exponential
    #   - exponential_staircase
    LEARNING_RATE_DECAY_TYPE = "exponential"
    # Number of steps the Learning rate should "linearly"
    # decay to MIN_LEARNING_RATE
    # For "exponential" decay, the learning rate is updated as
    # decayed_learning_rate = LEARNING_RATE *
    #                         LEARNING_RATE_DECAY ^ (step / LEARNING_RATE_DECAY_STEPS)
    # with integer dvision for "exponential_staircase"
    LEARNING_RATE_DECAY_STEPS = 200
    # Only has an effect for "exponential" decay
    LEARNING_RATE_DECAY = 0.97
    # Minimal Learning Rate
    MIN_LEARNING_RATE = 0.000001


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=int,
                        help='The ID for the current learnin run')
    parser.add_argument('glances', type=int,
                        help='The number of glances that are perceived per training-step')
    parser.add_argument('--heatmaps', type=bool, default=False, help='Create Heatmaps of the sensor policy')
    parser.add_argument('--sensor_policy', type=bool, default=True, help='Save sensor policy')
    parser.add_argument('--random', type=bool, default=False, help='Use a randomly generated sensor policy')
    parser.add_argument('--core', type=str, default='LSTM', help='Which kind of core unit should be used? LSTM or MLP')
    args = parser.parse_args()
    print("Training ID: {}\nNumber of Glances: {}\nCore Module: {}".format(args.id, args.glances, args.core))
    params = PARAMETERS
    params.GLANCES = args.glances
    params.SAVE_SENSOR_POLICY = args.sensor_policy
    params.HEATMAPS = args.heatmaps
    params.RANDOM_LOCS = args.random
    params.CORE_NET = args.core

    if params.RANDOM_LOCS:
        params.RESULTS_PATH = './Results_RANDOM_' + str(params.CORE_NET) + '_' + str(params.GLANCES) \
                              + 'Glances/' + str(args.id) + '/'
    else:
        params.RESULTS_PATH = './Results_' + str(params.CORE_NET) + '_' + str(params.GLANCES) \
                       + 'Glances/' + str(args.id) + '/'
    params.MODEL_FILE_PATH = params.RESULTS_PATH + 'Model/'
    if params.VISUALIZATION:
        from experiment_visualize import Experiment
    else:
        from experiment import Experiment

    Experiment(params)

if __name__ == '__main__':
    main()
