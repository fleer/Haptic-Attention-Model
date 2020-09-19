# coding=utf-8
# In order to use tf.print
from __future__ import print_function
import tensorflow as tf

class RAM():
    """
    Neural Network class that uses Tensorflow to build and train the Haptic Attention Model
    """


    def __init__(self, PARAMETERS, session):
        """
        Intialize parameters, determine the learning rate decay and build the RAM
        :param PARAMETERS: The Class with the used parameters
        :param session: The tensorflow session
        """

        self.session = session

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
        self.random_locs = tf.compat.v1.placeholder(tf.bool, shape=())

        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=())

        scope = "coreNetwork"
        # Global weights
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            # Initialize weights of location network
            self.h_l_out_x = tf.keras.layers.Dense(1, activation='tanh',
                    kernel_initializer = tf.keras.initializers.HeNormal(), name= "m_x")
            self.h_l_std_out_x = tf.keras.layers.Dense(1, activation='sigmoid',
                    kernel_initializer = tf.keras.initializers.HeNormal(), name= "std_x")
            self.h_l_out_a = tf.keras.layers.Dense(1, activation='tanh',
                    kernel_initializer = tf.keras.initializers.HeNormal(), name= "m_a")
            self.h_l_std_out_a = tf.keras.layers.Dense(1, activation='sigmoid',
                    kernel_initializer = tf.keras.initializers.HeNormal(), name= "std_a")
            self.b_l_out = tf.keras.layers.Dense(1, kernel_initializer = tf.keras.initializers.HeNormal())

            # Initialize weights of action network
            self.a_h_out = tf.keras.layers.Dense(self.output_dim, kernel_initializer = tf.keras.initializers.HeNormal())

            # Initialize weights of haptic network
            self.hn_1 = tf.keras.layers.Dense(self.hs_size, activation='relu',
                    kernel_initializer = tf.keras.initializers.HeNormal())

            self.glimpse_hg = tf.keras.layers.Dense(self.hn_size, activation='relu',
                    kernel_initializer = tf.keras.initializers.HeNormal())
            self.l_hl = tf.keras.layers.Dense(self.hs_size, activation='relu',
                    kernel_initializer = tf.keras.initializers.HeNormal())

            # For concatenation
            self.hg_1 = tf.keras.layers.Dense(self.hs_size, activation='relu',
                    kernel_initializer = tf.keras.initializers.HeNormal())
            self.hg_2 = tf.keras.layers.Dense(self.hs_size, activation='relu',
                    kernel_initializer = tf.keras.initializers.HeNormal())

            # Create LSTM Cell
            self.lstm_cell_1 = tf.keras.layers.LSTMCell(self.hs_size,
                    activation="relu", name="cell_1")
            # self.lstm_cell_1 = tf.compat.v1.nn.rnn_cell.LSTMCell(self.hs_size,
            #         activation="relu", state_is_tuple=True, name="cell_1")

            # Tensorflow Placeholder
            self.actions = tf.compat.v1.placeholder(shape=(self.batch_size,), dtype=tf.int64)
            self.actions_onehot = tf.one_hot(self.actions, self.output_dim, dtype=tf.float32)
            self.input_hs_1 = tf.compat.v1.placeholder(tf.float32, shape=(2, self.batch_size, self.hs_size))
            self.glances = tf.compat.v1.placeholder(shape=(), dtype=tf.float32)
            self.loss_list_b = tf.compat.v1.placeholder(shape=(), dtype=tf.float32)
            self.input_pressure= tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size, self.sensor_size), name="pressure")
            self.input_location= tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size,2), name="location")
            self.input_mean_x = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size,1), name="mean_x")
            self.input_mean_a = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size,1), name="mean_a")
            self.input_std_x = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size,1), name="std_x")
            self.input_std_a = tf.compat.v1.placeholder(tf.float32, shape=(self.batch_size,1), name="std_a")

            # Initial Generation of the sensor Location and initialization of the LSTM
            self.initial_loc_x, self.initial_loc_a, self.initial_mean_x, self.initial_mean_a, self.initial_std_x, \
            self.initial_std_a, self.init_state_1 = self.initial_touch()

            # Execute a glance
            self.loc_x, self.loc_a, self.mean_x, self.mean_a, self.std_x, self.std_a, self.state_1, \
            self.predicted_probs, baseline = self.glance(scope)

            # Compute the Loss
            self.cost_a, self.cost_l, self.cost_s, self.cost_b, self.reward, \
            all_grads = self.loss(baseline, self.mean_x, self.mean_a, self.std_x, self.std_a, scope)

            # Train
            self.train(all_grads, scope)

            # Compute number of trainable variables
            self.all_trainable_vars = tf.reduce_sum(
                    [tf.reduce_prod(v.shape) for v in tf.compat.v1.trainable_variables()])


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
            trainer = tf.keras.optimizers.RMSProp(learning_rate=self.learning_rate)
        elif self.optimizer == "adam":
            trainer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == "adadelta":
            trainer = tf.keras.optimizers.Adadelta(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            trainer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum, nesterov=True)
        else:
            raise ValueError("unrecognized update: {}".format(self.optimizer))

        local_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
        self.accum_vars = [tf.Variable(tf.zeros_like(tv.read_value()), trainable=False) for tv in local_vars]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]

        # Apply local gradients to global network
        ## Adds to each element from the list you initialized earlier with zeros its gradient
        # (works because accum_vars and gvs are in the same order)

        limit = tf.equal(self.glances, self.all_glances-1)
        # limit = tf.compat.v1.Print(limit, [limit])
        self.accum_ops = tf.cond(limit,
             lambda: [[self.accum_vars[i].assign_add(gv) for i, gv in enumerate(grads)]
                      for grads in all_grads], lambda: [[0. for i, gv in enumerate(grads)]
                                                        for grads in all_grads])
        self.apply_grads = trainer.apply_gradients(zip(self.accum_vars, local_vars))

    def initial_touch(self):
        """
        Initialize the variables and generate
        a first location-orientation pair for the sensor
        :return: the generated means, standard deviations, sampled location, sampled orientation
        and the initialized internal state of the LSTM
        """
        initial_state_1 = tf.zeros((2, self.batch_size, self.hs_size))
        # Initial location mean generated by initial hidden state of RNN
        mean_x = tf.random.uniform((self.batch_size,1), minval=-1., maxval=1.)
        std_x = tf.random.uniform((self.batch_size, 1), minval=0., maxval=1.)
        mean_a = tf.random.uniform((self.batch_size,1), minval=-1., maxval=1.)
        std_a = tf.random.uniform((self.batch_size, 1), minval=0., maxval=1.)
        loc_a = mean_a + tf.random.normal(mean_a.get_shape(), 0, std_a)
        loc_x = mean_x + tf.random.normal(mean_x.get_shape(), 0, std_x)
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
            output_1 = self.hn_1(h_feedback)
        elif self.core_net == "LSTM":
            state_1 = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(state_1[0], state_1[1])
            output_1, state_1 = self.lstm_cell_1(h_feedback, state_1)
        else:
            import sys
            print("Wrong option for core network: {}".format(self.core_net))
            sys.exit(0)

        # Location mean generated by initial hidden state of RNN
        mean_x = tf.cond(self.random_locs,
                                lambda: tf.random.uniform((self.batch_size,1), minval=-1., maxval=1.),
                                lambda: self.h_l_out_x(output_1))
        mean_a = tf.cond(self.random_locs,
                                lambda: tf.random.uniform((self.batch_size,1), minval=-1., maxval=1.),
                                lambda: self.h_l_out_a(output_1))
        std_x = tf.cond(self.random_locs,
                                lambda: tf.random.uniform((self.batch_size,1), minval=0., maxval=1.),
                                lambda: self.h_l_std_out_x(output_1))
        std_a = tf.cond(self.random_locs,
                                lambda: tf.random.uniform((self.batch_size,1), minval=0., maxval=1.),
                                lambda: self.h_l_std_out_a(output_1))
        loc_a = mean_a + tf.random.normal(mean_a.get_shape(), 0, std_a)
        loc_x = mean_x + tf.random.normal(mean_x.get_shape(), 0, std_x)

        # look at ONLY THE END of the sequence to predict label
        action_out = tf.nn.log_softmax(self.a_h_out(tf.reshape(output_1, (self.batch_size, self.hs_size))))

        baseline = self.b_l_out(tf.reshape(output_1, (self.batch_size, self.hs_size)))

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

        max_p_y = tf.argmax(self.predicted_probs, axis=-1)
        R_max= tf.cast(tf.equal(max_p_y, self.actions), tf.float32)
        R_max = tf.stop_gradient(R_max)
        R = tf.reshape(R_max,[self.batch_size,1])

        # reward per example
        # mean reward
        reward = tf.reduce_mean(R_max)

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

        b_ng = tf.stop_gradient(baseline)

        # The Loss is computed on the data of the locations/orientations of the "previous step"!
        # They are just inputs and not computed by the network.
        # In order to assure that the loss is nevertheless piped backwards correctly, the new computed locatuions/orientations are added and multiplied with 0
        # like + tf.concat([mean_x, mean_a], axis=-1) * 0

        Reinforce = (self.input_location - tf.concat([self.input_mean_x, self.input_mean_a], axis=-1))/(tf.concat([self.input_std_x, self.input_std_a], axis=-1)**2)\
                    * (R-b_ng) \
                    + tf.concat([mean_x, mean_a], axis=-1) * 0
        Reinforce_std = (((self.input_location - tf.concat([self.input_mean_x, self.input_mean_a], axis=-1))**2)-tf.concat([self.input_std_x, self.input_std_a], axis=-1)**2)/(tf.concat([self.input_std_x, self.input_std_a], axis=-1)**3) \
                        * (R-b_ng)+ tf.concat([std_x,std_a], axis=-1) * 0
        Reinforce = tf.reduce_sum(Reinforce, axis=-1)
        Reinforce_std = tf.reduce_sum(Reinforce_std, axis=-1)

        J = tf.reduce_sum(self.predicted_probs * self.actions_onehot, axis=1)

        # Hybrid Loss
        # balances the scale of the two gradient components
        cost = - tf.reduce_mean(J + self.location_weight * (Reinforce + Reinforce_std), axis=0)# * (R[-1]-b_ng[-1])
        cost_R = - tf.reduce_mean(J, axis=0)

        cost = tf.cond(self.random_locs, lambda: cost_R, lambda: cost)

        # Baseline is trained with MSE
        b_loss = tf.losses.mean_squared_error(R, baseline) + self.loss_list_b
        b_loss = tf.cond(self.random_locs, lambda: tf.zeros_like(b_loss, dtype=tf.float32), lambda: b_loss)

        local_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
        # TODO: Implement gradient clipping
        all_grads = []
        all_grads_no_zero = []
        all_grads.append(tf.compat.v1.gradients(cost, local_vars))
        all_grads.append(tf.compat.v1.gradients(b_loss/self.glances, local_vars))
        for grads in all_grads:
           all_grads_no_zero.append([grad if grad is not None else tf.zeros_like(var) for grad, var in zip(grads, local_vars)])
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
        hg = self.glimpse_hg(pressure)
        # Process locations
        hl = self.l_hl(loc)
        #Combine the glimpses via concatenation
        concat = tf.concat([hg,hl], axis=-1)
        g_1 = self.hg_1(concat)
        g = self.hg_2(g_1)

        return g
