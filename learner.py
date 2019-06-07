from network import RAM
import numpy as np
import h5py
from collections import defaultdict
import sys
import logging

#import rospy
#from tactile_msgs.msg import TactileState
#from std_msgs.msg import String

class Learner():

    def __init__(self,PARAMETERS, sess):
        """
        Initialize the Learner class
        :param PARAMETERS: The class with all specified parameters for the current learning run
        :param sess: tensorflow session
        """

        self.session = sess

        self.logger = logging.getLogger("Learner")
        #self.logger.setLevel(level=logging.DEBUG)
        self.logger.setLevel(level=logging.INFO)
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        #ch.setLevel(logging.DEBUG)
        ch.setLevel(logging.INFO)
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: >>> %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        self.logger.addHandler(ch)

        #   ================
        #   Creating the RAM
        #   ================
        self.ram = RAM(PARAMETERS, sess)
        self.logger.info("Trainable Weights: {}".format(sess.run(self.ram.all_trainable_vars)))

        #   =====================
        #   Define some variables
        #   =====================
        self.batch_size = PARAMETERS.BATCH_SIZE
        self.glances = PARAMETERS.GLANCES
        self.sensor_size = 256
        self.num_objects = PARAMETERS.OBJECT_NUM
        if self.glances == 1:
            self.random_locs = 1
        else:
            self.random_locs = int(PARAMETERS.RANDOM_LOCS)

        # Learning Rate
        self.max_lr = PARAMETERS.LEARNING_RATE
        self.min_lr = PARAMETERS.MIN_LEARNING_RATE
        self.lr_decay_rate = PARAMETERS.LEARNING_RATE_DECAY
        self.lr_decay_steps = PARAMETERS.LEARNING_RATE_DECAY_STEPS
        self.lr_decay_type = PARAMETERS.LEARNING_RATE_DECAY_TYPE
        self.lr = PARAMETERS.LEARNING_RATE
        self.step = 0

        self.gazebo = PARAMETERS.TRAIN_GAZEBO or PARAMETERS.VISUALIZATION
        # Learning Rate Decay
        if self.lr_decay_steps != 0 and self.lr_decay_type == "linear":
            self.lr_decay_rate = ((self.lr - self.min_lr) /
                                  self.lr_decay_steps)


        #   =====================
        #   Read in the data
        #   =====================

        # Initialize gazebo interface
        if self.gazebo:
            rospy.init_node('learner', anonymous=True)
            rospy.Subscriber("/network/myrmex", TactileState, self.callback)
            self.pub = rospy.Publisher('/network/pose', String, queue_size=1)


        else:
           self.training_data = h5py.File("./objects.hdf5","r")

    def callback(self, data):
        """
        Callback for receiving the sensory data from the Gazebo simulation
        :param data: the gathered sensory
        :return:
        """
        self.taxels = data.sensors[0]
        self.recieved_data = True


    def create_batch(self, test=False):
        """
        Creates a batch of data
        :param test: if True, than all objects are tested equally often
        It is assumed that batch-size/num_objects is an integer
        :return: Label X
        """
        X = []
        objects = np.arange(self.num_objects, dtype=int)
        if test:
            for p in range(self.num_objects):
                for _ in range(int(self.batch_size/self.num_objects)):
                    X.append(p)
        else:
            for _ in range(self.batch_size):
                object = np.random.choice(objects)
                X.append(object)
        assert len(X) == self.batch_size, "Wrong length of Batch"
        return X


    def visualize(self, object):
        """
        Train the network
        :param object: the ID of the to-be-identified object
        :return: Mean reward, predicted labels, accumulated loss, location policy loss, baseline loss
        """
        # Compute initial pose, using hidden state of LSTM 1
        fetches = [self.ram.initial_loc_x, self.ram.initial_loc_a, self.ram.initial_mean_x, self.ram.initial_mean_a,
                   self.ram.initial_std_x, self.ram.initial_std_a, self.ram.init_state_1]
        loc_x, loc_a, mean_x, mean_a, std_x, std_a, state_1, = self.session.run(fetches=fetches)

        # Swipes
        for g in range(self.glances):
            # Foreward Pass
            # If location or orientation are out of range, resample it
            for z in range(len(loc_x)):
                emergency_count = 0
                while abs(loc_x[z]) > 1.:
                    assert emergency_count < 1000, "Caught in Loop for cutting the new location for position! Mean: {}, Std: {}".format(mean_x[z], std_x[z])
                    loc_x[z] = np.random.normal(mean_x[z], std_x[z])
                    emergency_count += 1
                emergency_count = 0
                while abs(loc_a[z]) > 1.:
                    assert emergency_count < 1000, "Caught in Loop for cutting the new location for angle! Mean: {}, Std: {}".format(mean_a[z], std_a[z])
                    loc_a[z] = np.random.normal(mean_a[z], std_a[z])
                    emergency_count += 1
            # Get corresponding pressure
            pressure = self.get_pressure_gazebo(loc_x, loc_a, object)
            loc = np.concatenate([loc_x, loc_a], axis=-1)
            feed_dict={self.ram.input_pressure: [pressure], self.ram.input_location: loc, self.ram.random_locs: self.random_locs,
                       self.ram.input_hs_1: state_1,
                       self.ram.actions: [object]}
            fetches = [self.ram.loc_x, self.ram.loc_a, self.ram.mean_x, self.ram.mean_a, self.ram.std_x,
                       self.ram.std_a, self.ram.reward, self.ram.predicted_probs, self.ram.state_1]
            loc_x, loc_a, mean_x, mean_a, std_x, std_a, glance_reward, glance_action, \
            state_1 = self.session.run(feed_dict=feed_dict, fetches=fetches)

        print ("Log_Softmax output: {}".format(glance_action))
        p_max = np.argmax(glance_action, axis=-1)
        print ("Predicted object: {}".format(p_max[0]))
        correct = np.equal(p_max[0], object)
        print ("Network Reward: " + str(glance_reward))

        return p_max, correct

    def evaluate(self):
        """
        Evaluate the network
        :return: mean accuracy, standard deviation of the mean of the accuracy,
        a list of the correct prediction per glance, a list of the accumulated prediction per glance,
        list of locations, lists of means and lists of standard deviations
        """
        location_save = defaultdict(list)
        location_save_mu_std_x = defaultdict(list)
        location_save_mu_std_a = defaultdict(list)
        location_save_mu_x_a = defaultdict(list)

        accumulated_glance_reward_list = np.zeros(self.glances)
        glance_reward_list = defaultdict(list)
        reward_list = []

        num_test_batches = 100

        if self.gazebo == True:
            num_test_batches = 10

        for i in range(num_test_batches):
            # Create new batch
            # X: Object IDs
            X = self.create_batch(test=True)

            self.logger.debug("Testbatch = " + str(i))
            for index, x in enumerate(X):
                self.logger.debug("Object: " + str(x))
                # Compute initial pose, using hidden state of LSTM 1
                fetches = [self.ram.initial_loc_x, self.ram.initial_loc_a, self.ram.initial_mean_x, self.ram.initial_mean_a,
                           self.ram.initial_std_x, self.ram.initial_std_a, self.ram.init_state_1]
                loc_x, loc_a, mean_x, mean_a, std_x, std_a, state_1 = self.session.run(fetches=fetches)

                glance_actions = []
                # Swipes
                for g in range(self.glances):
                    # Foreward Pass
                    for z in range(len(loc_x)):
                        emergency_count = 0
                        while abs(loc_x[z]) > 1.:
                            assert emergency_count < 1000, "Caught in Loop for cutting the new location for position! Mean: {}, Std: {}".format(mean_x[z], std_x[z])
                            loc_x[z] = np.random.normal(mean_x[z], std_x[z])
                            emergency_count += 1
                        emergency_count = 0
                        while abs(loc_a[z]) > 1.:
                            assert emergency_count < 1000, "Caught in Loop for cutting the new location for angle! Mean: {}, Std: {}".format(mean_a[z], std_a[z])
                            loc_a[z] = np.random.normal(mean_a[z], std_a[z])
                            emergency_count += 1
                    mean_x_old = mean_x
                    mean_a_old = mean_a
                    std_x_old = std_x
                    std_a_old = std_a

                    # Get corresponding pressure
                    if self.gazebo:
                        pressure = self.get_pressure_gazebo(loc_x, loc_a, x)
                        self.logger.debug("Swipe: 0")
                    else:
                        pressure = self.get_pressure(loc_x, loc_a, x)

                    loc = np.concatenate([loc_x, loc_a], axis=-1)

                    feed_dict={self.ram.input_pressure: [pressure], self.ram.input_location: loc,
                               self.ram.random_locs: self.random_locs,
                               self.ram.input_hs_1: state_1,
                               self.ram.actions: [x]}
                    fetches = [self.ram.loc_x, self.ram.loc_a, self.ram.mean_x, self.ram.mean_a, self.ram.std_x,
                               self.ram.std_a, self.ram.reward, self.ram.predicted_probs, self.ram.state_1]
                    loc_x, loc_a, mean_x, mean_a, std_x, std_a, glance_reward, glance_action, \
                    state_1 = self.session.run(feed_dict=feed_dict, fetches=fetches)

                    glance_actions.append(glance_action)
                    # Computing the reward when using the accumulated accuracy of the previous glances
                    accumulated_glance_reward_list[g] += np.equal(
                        np.argmax(np.mean(glance_actions, axis=0), axis=-1), x).astype(float)


                reward_list.append(glance_reward)

                location_save[x].extend(loc)
                location_save_mu_x_a[x].extend(np.concatenate([mean_x_old,mean_a_old], axis=-1))
                location_save_mu_std_x[x].extend(np.concatenate([mean_x_old,std_x_old], axis=-1))
                location_save_mu_std_a[x].extend(np.concatenate([mean_a_old,std_a_old], axis=-1))

        reward_sum_sqrt = np.sum(np.square(reward_list))
        num_data = len(reward_list)
        accuracy = np.mean(reward_list)
        accuracy_std = np.sqrt(((reward_sum_sqrt/num_data) - accuracy**2)/num_data)

        return accuracy, accuracy_std, glance_reward_list, \
               accumulated_glance_reward_list/float(self.batch_size*num_test_batches), location_save, \
                location_save_mu_x_a, location_save_mu_std_x, location_save_mu_std_a

    def train(self):
        """
        Train the network
        :return: Mean reward, predicted labels, accumulated loss, REINFORCE mean, REINFORCE std, baseline loss,
        list of reward per glance, list of accumulated reward per glance
        """
        # store glance reward for evaluation
        reward_list = np.zeros((self.glances, self.batch_size))
        accumulated_reward_list = np.zeros(self.glances)
        batch_reward = []
        batch_predicted_labels = []
        batch_cost_a = []
        batch_cost_l = []
        batch_cost_s = []
        batch_cost_b = []

        # Create new batch
        # X : Object IDs
        # Y; Target/Distractor
        X = self.create_batch()
        self.logger.debug("Train Batch")
        self.session.run(self.ram.zero_ops)
        for index, x in enumerate(X):
            self.logger.debug("Object:" + str(x))
            cost_b_fetched = 0

            # Compute initial pose, using hidden state of LSTM 1
            fetches = [self.ram.initial_loc_x, self.ram.initial_loc_a, self.ram.initial_mean_x, self.ram.initial_mean_a,
                       self.ram.initial_std_x, self.ram.initial_std_a, self.ram.init_state_1]
            loc_x, loc_a, mean_x, mean_a, std_x, std_a, state_1 = self.session.run(fetches=fetches)

            glance_actions = []
            # Swipes
            for g in range(self.glances):
                # Foreward Pass
                # Classify and train on the collected data
                # Compute initial pose, using hidden state of LSTM 1

                for z in range(len(loc_x)):
                    emergency_count = 0
                    while abs(loc_x[z]) > 1.:
                        assert  emergency_count < 1000, "Caught in Loop for cutting the new location for position! Mean: {}, Std: {}".format(mean_x[z], std_x[z])
                        loc_x[z] = np.random.normal(mean_x[z], std_x[z])
                        emergency_count += 1
                    emergency_count = 0
                    while abs(loc_a[z]) > 1.:
                        assert  emergency_count < 1000, "Caught in Loop for cutting the new location for angle! Mean: {}, Std: {}".format(mean_a[z], std_a[z])
                        loc_a[z] = np.random.normal(mean_a[z], std_a[z])
                        emergency_count += 1
                # Get corresponding pressure
                if self.gazebo:
                    pressure = self.get_pressure_gazebo(loc_x, loc_a, x)
                    self.logger.debug("Swipe: 0")
                else:
                    pressure = self.get_pressure(loc_x,loc_a, x)

                loc = np.concatenate([loc_x, loc_a], axis=-1)

                feed_dict = {self.ram.input_pressure: [pressure],
                             self.ram.input_location: loc,
                             self.ram.input_hs_1: state_1,
                             self.ram.input_mean_x: mean_x,
                             self.ram.input_mean_a: mean_a,
                             self.ram.input_std_x: std_x,
                             self.ram.input_std_a: std_a,
                             self.ram.actions: [x],
                             self.ram.random_locs: self.random_locs,
                             self.ram.glances: g,
                             self.ram.loss_list_b: cost_b_fetched,
                             self.ram.learning_rate: self.lr}
                fetches = [self.ram.loc_x,
                           self.ram.loc_a,
                           self.ram.mean_x,
                           self.ram.mean_a,
                           self.ram.std_x,
                           self.ram.std_a,
                           self.ram.state_1,
                           self.ram.cost_a,
                           self.ram.cost_l,
                           self.ram.cost_s,
                           self.ram.cost_b,
                           self.ram.reward,
                           self.ram.predicted_probs,
                           self.ram.accum_ops]

                loc_x, loc_a, mean_x, mean_a, std_x, std_a, \
                state_1, cost_a_fetched, cost_l_fetched, cost_s_fetched, \
                cost_b_fetched, reward_fetched, prediction_labels_fetched, _ \
                    = self.session.run(feed_dict=feed_dict, fetches=fetches)
                reward_list[g][index] = reward_fetched

                glance_actions.append(prediction_labels_fetched[0])
                # Computing the reward when using the accumulated accuracy of the previous glances
                accumulated_reward_list[g] += np.equal(
                        np.argmax(np.mean(glance_actions, axis=0), axis=-1), x).astype(float)


            # Save the data of the current batch
            batch_reward.append(reward_fetched)
            batch_predicted_labels.append(prediction_labels_fetched)
            batch_cost_a.append(cost_a_fetched)
            batch_cost_l.append(cost_l_fetched)
            batch_cost_s.append(cost_s_fetched)
            batch_cost_b.append(cost_b_fetched/(g+1.))
        # Here, the action training happens by applying the gradients
        self.session.run(self.ram.apply_grads, feed_dict={self.ram.learning_rate: self.lr})

        return batch_reward, batch_predicted_labels, batch_cost_a, batch_cost_l, batch_cost_s, \
               batch_cost_b, reward_list, accumulated_reward_list/float(self.batch_size)

    def get_pressure(self, pos, eul, obj_id):
        """
        Get the pressure from the pre-recorded data
        :param pos: batch of positions in the range [-1,1]
        :param eul: batch of orientations in the range [-1,1]
        :param obj_id: batch of the corresponding object ids
        :return: batch of normalized pressures
        """
        eul = np.squeeze(eul)
        pos = np.squeeze(pos)
        assert abs(pos) <= 1., "Position not in range [-1,1] --> {}".format(pos)
        assert abs(eul) <= 1., "Angle not in range [-1,1] --> {}".format(eul)
        d = self.training_data["object_" + str(obj_id)]

        index_l = int((1+pos) * (len(d[0])-1)/2)
        index_e = int((1+eul) * (len(d[0])-1)/2)
        pressure = d[index_l][index_e]
        return pressure

    def translate_pose(self, pose, object):
        """
        Translates the pose [-1,1] into the zone of the desired object
        :param pose:
        :param object:
        :return:
        """
        assert isinstance(object, int)
        tp = ((pose + 1.) / 2.)
        if object == 0:
            return 0.12 + tp * 0.18
        elif object == 1:
            return 0.34 + tp * 0.13
        elif object == 2:
            return 0.49 + tp * 0.14
        elif object == 3:
            return 0.67 + tp * 0.11
        else:
            print("Wrong object identifier: {}".format(object))
            sys.exit(0)

    def get_pressure_gazebo(self,pos, eul, obj_id):
        """
        Get the pressure from the gazebo simulation
        :param pos: position in the range [-1,1]
        :param eul: orientation in the range [-1,1]
        :param obj_id: the corresponding object id
        :return: normalized pressures
        """
        rate = rospy.Rate(10) # 10hz
        assert abs(pos) <= 1. + 1e-6, "Position not in range [-1,1] --> {}".format(pos)
        assert abs(eul) <= 1. + 1e-6, "Angle not in range [-1,1] --> {}".format(eul)
        eul = np.squeeze(eul)
        pos = np.squeeze(pos)
        orientation = np.math.pi * float(eul) * 0.3
        position = self.translate_pose(float(pos), obj_id)
        position = position - 0.068 * np.sin(orientation)
        self.recieved_data = False
        hello_str = str(position) + " " + str(orientation)
        while self.recieved_data == False:
            self.pub.publish(hello_str)
            rate.sleep()

        m = str(self.taxels)
        m = m[27:]
        m = m.split(', ', 256)
        m[-1] = m[-1][:-1]
        m = np.asarray(m, dtype=float)
        assert len(m) == 256, "Wrong length of pressure Array --> {}".format(len(m))
        #  print(np.sum(m))
        if np.sum(m) > 0:
            pressure = m/np.linalg.norm(m, ord=1)
            assert np.sum(pressure) - 1. <= 1e-6, "Pressure should be normalized! --> {}".format(np.sum(pressure))
        else:
            pressure = m
        #  print(np.sum(pressure))
        return pressure

    def learning_rate_decay(self):
        """
        Function to control the linear decay
        of the learning rate
        :return: New learning rate
        """
        if self.lr_decay_type == "static":
            return self.lr
        elif self.lr_decay_type == "linear":
            # Linear Learning Rate Decay
            self.lr = max(self.min_lr, self.lr - self.lr_decay_rate)
        elif self.lr_decay_type == "exponential":
            # Exponential Learning Rate Decay
            self.lr = max(self.min_lr, self.max_lr * (self.lr_decay_rate ** (self.step/self.lr_decay_steps)))
        elif self.lr_decay_type == "exponential_staircase":
            # Exponential Learning Rate Decay
            self.lr = max(self.min_lr, self.max_lr * (self.lr_decay_rate ** (self.step // self.lr_decay_steps)))
        else:
            print("Wrong type of learning rate: " + self.lr_decay_type)
            return 0
        self.step += 1

        return self.lr
