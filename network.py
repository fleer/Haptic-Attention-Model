# coding=utf-8
# In order to use tf.print
from __future__ import print_function
import tensorflow as tf

class RAM(tf.keras.Model):
    """
    Neural Network class that uses Tensorflow to build and train the Haptic Attention Model
    """


    def __init__(self, PARAMETERS):
        """
        Intialize parameters, determine the learning rate decay and build the RAM
        :param PARAMETERS: The Class with the used parameters
        :param session: The tensorflow session
        """

        super(RAM, self).__init__()

        # number of classes
        self.output_dim = PARAMETERS.OBJECT_NUM
        self.batch_size = 1
        self.real_batch_size = float(PARAMETERS.BATCH_SIZE)
        self.all_glances= PARAMETERS.GLANCES
        self.sensor_size = 256
        self.core_net = PARAMETERS.CORE_NET


        # Learning
        self.optimizer = PARAMETERS.OPTIMIZER
        self.momentum = PARAMETERS.MOMENTUM

        self.glimpses_list = []
        self.state_1 = []

        # Size of Hidden state
        self.hs_size = PARAMETERS.HIDDEN_STATE
        self.hn_size = PARAMETERS.HAPTIC_NET
        self.location_weight = PARAMETERS.LOCATION_WEIGHT

        # If random_locs --> True, Random Location Policy is used
        self.random_locs = tensorflow.placeholder(tensorflow.bool, shape=[])

        self.learning_rate = tensorflow.placeholder(tensorflow.float32, shape=[])

        # Initialize weights of location network
        self.h_l_out_x = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer = tf.keras.initializers.HeNormal())
        self.h_l_std_out_x = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer = tf.keras.initializers.HeNormal())
        self.h_l_out_a = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer = tf.keras.initializers.HeNormal())
        self.h_l_std_out_a = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer = tf.keras.initializers.HeNormal())
        self.b_l_out = tf.keras.layers.Dense(1, kernel_initializer = tf.keras.initializers.HeNormal())

        # Initialize weights of action network
        self.a_h_out = tf.keras.layers.Dense(self.output_dim,
                activation=tf.nn.log_softmax,
                kernel_initializer = tf.keras.initializers.HeNormal())

        # Initialize weights of glimpse network
        self.glimpse_hg = self.weight_variable((self.sensor_size,self.hn_size))
        self.l_hl = tf.keras.layers.Dense(2, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal())
        self.hg_g = tf.keras.layers.Dense(self.hn_size, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal())
        self.hl_g = tf.keras.layers.Dense(self.hn_size, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal())
        self.hn_1 = tf.keras.layers.Dense(self.hs_size, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal())

        # For concatenation
        self.hg_1 = tf.keras.layers.Dense(self.hn_size + self.hn_size, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal())
        self.hg_2 = tf.keras.layers.Dense(self.hs_size, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal())

        # Create LSTM Cell
        self.lstm_cell_1 = tf.keras.layers.LSTMCell(self, activation='relu')

    #    # Initial Generation of the sensor Location and initialization of the LSTM
    #    self.initial_loc_x, self.initial_loc_a, self.initial_mean_x, self.initial_mean_a, self.initial_std_x, \
    #    self.initial_std_a, self.init_state_1 = self.initial_touch()

    #    # Execute a glance
    #    self.loc_x, self.loc_a, self.mean_x, self.mean_a, self.std_x, self.std_a, self.state_1, \
    #    self.predicted_probs, baseline = self.glance(scope)

    #    # Compute the Loss
    #    self.cost_a, self.cost_l, self.cost_s, self.cost_b, self.reward, \
    #    all_grads = self.loss(baseline, self.mean_x, self.mean_a, self.std_x, self.std_a, scope)

    #    # Train
    #    self.train(all_grads, scope)

    #    # Compute number of trainable variables
    #    self.all_trainable_vars = tensorflow.reduce_sum(
    #            [tensorflow.reduce_prod(v.shape) for v in tensorflow.trainable_variables()])


    def train(self, all_grads, scope):
        """
        Training the model by manually accumulate the computed
        gradients of the last glance for all samples of the current batch
        :param all_grads: the list of computed gradients
        :param scope: used variable scope
        :return:
        """
        # Choose Optimizer
        if self.optimizer == "rmsprop":
            trainer = tensorflow.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == "adam":
            trainer = tensorflow.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == "adadelta":
            trainer = tensorflow.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            trainer = tensorflow.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum, use_nesterov=True)
        else:
            raise ValueError("unrecognized update: {}".format(self.optimizer))

        local_vars = tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES, scope)
        self.accum_vars = [tensorflow.Variable(tensorflow.zeros_like(tv.initialized_value()), trainable=False) for tv in local_vars]
        self.zero_ops = [tv.assign(tensorflow.zeros_like(tv)) for tv in self.accum_vars]

        # Apply local gradients to global network
        ## Adds to each element from the list you initialized earlier with zeros its gradient
        # (works because accum_vars and gvs are in the same order)

        limit = tensorflow.equal(self.glances, self.all_glances-1)
        #limit = tensorflow.Print(limit, [limit])
        self.accum_ops = tensorflow.cond(limit,
                                         lambda: [[self.accum_vars[i].assign_add(gv) for i, gv in enumerate(grads)]
                                                  for grads in all_grads], lambda: [[0. for i, gv in enumerate(grads)]
                                                                                    for grads in all_grads])
        self.apply_grads = trainer.apply_gradients(zip(self.accum_vars, local_vars))

    def weight_variable(self,shape, name=None):
        """
        Trainable network weights are initialized with uniform
        value within the range [-0.01, 0.01]
        he_uniform initialization
        References:
            He et al., http://arxiv.org/abs/1502.01852
        :param shape: Desired shape
        :return: Tensorflow variable
        """
        limit = tensorflow.sqrt(6. / tensorflow.cast(tensorflow.maximum(0, shape[0]), tensorflow.float32))
        if name is None:
            initial = tensorflow.random_uniform(shape, minval=-limit, maxval=limit)
        else:
            initial = tensorflow.random_uniform(shape, minval=-limit, maxval=limit, name = name)
        return tensorflow.Variable(initial)

    def initial_touch(self):
        """
        Initialize the variables and generate
        a first location-orientation pair for the sensor
        :return: the generated means, standard deviations, sampled location, sampled orientation
        and the initialized internal state of the LSTM
        """
        initial_state_1 = self.lstm_cell_1.zero_state(self.batch_size, tensorflow.float32)
        # Initial location mean generated by initial hidden state of RNN
        mean_x = tensorflow.random_uniform((self.batch_size,1), minval=-1., maxval=1.)
        std_x = tensorflow.random_uniform((self.batch_size, 1), minval=0., maxval=1.)
        mean_a = tensorflow.random_uniform((self.batch_size,1), minval=-1., maxval=1.)
        std_a = tensorflow.random_uniform((self.batch_size, 1), minval=0., maxval=1.)
        loc_a = mean_a + tensorflow.random_normal(mean_a.get_shape(), 0, std_a)
        loc_x = mean_x + tensorflow.random_normal(mean_x.get_shape(), 0, std_x)
        return loc_x, loc_a, mean_x, mean_a, std_x, std_a, initial_state_1

    def glance(self, scope):
        """
        The function for computing one single glance
        :param scope: the used variable scope
        :return: the generated means, standard deviations, sampled location, sampled orientation
        the predicted log_softmax output and the baseline
        """
        pressure = self.input_pressure
        loc = self.input_location
        h_feedback = self.Haptic_Net(pressure, loc)
        state_1 = self.input_hs_1

        if self.core_net == "MLP":
            output_1 = tensorflow.nn.relu(tensorflow.matmul(h_feedback, self.hn_1))
        elif self.core_net == "LSTM":
            state_1 = tensorflow.nn.rnn_cell.LSTMStateTuple(state_1[0], state_1[1])
            output_1, state_1 = self.lstm_cell_1(h_feedback, state_1)
        else:
            import sys
            print("Wrong option for core network: {}".format(self.core_net))
            sys.exit(0)

        # Location mean generated by initial hidden state of RNN
        mean_x = tensorflow.cond(self.random_locs,
                                 lambda: tensorflow.random_uniform((self.batch_size,1), minval=-1., maxval=1.),
                                 lambda: tensorflow.nn.tanh(tensorflow.matmul(output_1, self.h_l_out_x)))
        mean_a = tensorflow.cond(self.random_locs,
                                 lambda: tensorflow.random_uniform((self.batch_size,1), minval=-1., maxval=1.),
                                 lambda: tensorflow.nn.tanh(tensorflow.matmul(output_1, self.h_l_out_a)))
        std_x = tensorflow.cond(self.random_locs,
                                lambda: tensorflow.random_uniform((self.batch_size,1), minval=0., maxval=1.),
                                lambda: tensorflow.nn.sigmoid(tensorflow.matmul(output_1, self.h_l_std_out_x)))
        std_a = tensorflow.cond(self.random_locs,
                                lambda: tensorflow.random_uniform((self.batch_size,1), minval=0., maxval=1.),
                                lambda: tensorflow.nn.sigmoid(tensorflow.matmul(output_1, self.h_l_std_out_a)))
        loc_a = mean_a + tensorflow.random_normal(mean_a.get_shape(), 0, std_a)
        loc_x = mean_x + tensorflow.random_normal(mean_x.get_shape(), 0, std_x)

        # look at ONLY THE END of the sequence to predict label
        action_out = tensorflow.nn.log_softmax(
            tensorflow.matmul(tensorflow.reshape(output_1, (self.batch_size, self.hs_size)), self.a_h_out))

        baseline = tensorflow.matmul(tensorflow.reshape(output_1, (self.batch_size, self.hs_size)), self.b_l_out)

        return loc_x, loc_a, mean_x, mean_a, std_x, std_a, state_1, action_out, baseline


    def loss(self, baseline, mean_x, mean_a, std_x, std_a, scope):
        """
        Compute the loss functions and the reward
        :param baseline: predicted baseline
        :param mean_x: the generated mean of the location Gaussian
        :param mean_a: the generated mean of the orientation Gaussian
        :param std_x: the generated standard deviation of the location Gaussian
        :param std_a: the generated standard deviation of the location Gaussian
        :param scope: the used variable scope
        :return: the individual parts of the loss,
         the computed reward and a list of all non-zero gradients
        """

        # look at ONLY THE END of the sequence to predict label
        correct_y = tensorflow.cast(self.actions, tensorflow.int64)

        max_p_y = tensorflow.argmax(self.predicted_probs, axis=-1)
        R_max= tensorflow.cast(tensorflow.equal(max_p_y, correct_y), tensorflow.float32)
        R_max = tensorflow.stop_gradient(R_max)
        R = tensorflow.reshape(R_max,[self.batch_size,1])

        # reward per example
        # mean reward
        reward = tensorflow.reduce_mean(R_max)

        # REINFORCE algorithm for policy network loss
        # -------
        # Williams, Ronald J. "Simple statistical gradient-following
        # algorithms for connectionist reinforcement learning."
        # Machine learning 8.3-4 (1992): 229-256.
        # -------
        # characteristic eligibility taken from sec 6. p.237-239
        #
        # For mean:
        #
        # d ln(f(m,s,x))   (x - m)
        # -------------- = -------- with m = mean, x = sample, s = standard deviation
        #       d m          s**2
        #
        # For standard deviation:
        #
        # d ln(f(m,s,x))   (x - m)**2 - s**2
        # -------------- = ------------------ with m = mean, x = sample, s = standard deviation
        #       d s               s**3
        #

        b_ng = tensorflow.stop_gradient(baseline)

        # The Loss is computed on the data of the locations/orientations of the "previous step"!
        # They are just inputs and not computed by the network.
        # In order to assure that the loss is nevertheless piped backwards correctly, the new computed locatuions/orientations are added and multiplied with 0
        # like + tensorflow.concat([mean_x, mean_a], axis=-1) * 0

        Reinforce = (self.input_location - tensorflow.concat([self.input_mean_x, self.input_mean_a], axis=-1))/(tensorflow.concat([self.input_std_x, self.input_std_a], axis=-1)**2)\
                    * (R-b_ng) \
                    + tensorflow.concat([mean_x, mean_a], axis=-1) * 0
        Reinforce_std = (((self.input_location - tensorflow.concat([self.input_mean_x, self.input_mean_a], axis=-1))**2)-tensorflow.concat([self.input_std_x, self.input_std_a], axis=-1)**2)/(tensorflow.concat([self.input_std_x, self.input_std_a], axis=-1)**3) \
                        * (R-b_ng)+ tensorflow.concat([std_x,std_a], axis=-1) * 0
        Reinforce = tensorflow.reduce_sum(Reinforce, axis=-1)
        Reinforce_std = tensorflow.reduce_sum(Reinforce_std, axis=-1)

        J = tensorflow.reduce_sum(self.predicted_probs * self.actions_onehot, axis=1)

        # Hybrid Loss
        # balances the scale of the two gradient components
        cost = - tensorflow.reduce_mean(J + self.location_weight * (Reinforce + Reinforce_std), axis=0)# * (R[-1]-b_ng[-1])
        cost_R = - tensorflow.reduce_mean(J, axis=0)

        cost = tensorflow.cond(self.random_locs, lambda: cost_R, lambda: cost)

        # Baseline is trained with MSE
        b_loss = (tensorflow.losses.mean_squared_error(R, baseline) + self.loss_list_b)
        b_loss = tensorflow.cond(self.random_locs, lambda: tensorflow.constant(0., shape=b_loss.get_shape(),dtype=tensorflow.float32), lambda: b_loss)

        local_vars = tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES, scope)
        # TODO: Implement gradient clipping
        all_grads = []
        all_grads_no_zero = []
        all_grads.append(tensorflow.gradients(cost, local_vars))
        all_grads.append(tensorflow.gradients(b_loss/self.glances, local_vars))
        for grads in all_grads:
           all_grads_no_zero.append([grad if grad is not None else tensorflow.zeros_like(var) for var, grad in zip(local_vars, grads)])
        return cost, Reinforce, Reinforce_std, b_loss, reward, all_grads_no_zero

    def Haptic_Net(self, pressure, loc):
        """
        The 'Haptic Net' for combining the pressure
        information with the location of the sensor
        :param pressure: a pressure vector
        :param loc: a location
        :return: a feature vector
        """

        # Process pressure
        hg = tensorflow.nn.relu(tensorflow.matmul(pressure, self.glimpse_hg))
        # Process locations
        hl = tensorflow.nn.relu(tensorflow.matmul(loc, self.l_hl))
        #Combine the glimpses via concatenation
        concat = tensorflow.concat([hg,hl], axis=-1)
        g_1 = tensorflow.nn.relu(tensorflow.matmul(concat, self.hg_1))
        g = tensorflow.nn.relu(tensorflow.matmul(g_1, self.hg_2))

        return g
