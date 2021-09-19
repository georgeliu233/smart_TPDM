import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

layers = tf.keras.layers


class GaussianActor(tf.keras.Model):
    LOG_STD_CAP_MAX = 2  # np.e**2 = 7.389
    LOG_STD_CAP_MIN = -20  # np.e**-10 = 4.540e-05
    EPS = 1e-6

    def __init__(self, state_shape, action_dim, max_action, units=[256]*1,
                 hidden_activation="relu", state_independent_std=False,
                 squash=False, name='gaussian_policy',state_input=False,residual=False,lstm=False):
        super().__init__(name=name)

        self._state_independent_std = state_independent_std
        self.lstm = lstm
        self._squash = squash
        self.residual = residual

        # self.base_layers = []
        # for unit in units:
        #     self.base_layers.append(layers.Dense(unit, activation=hidden_activation))
        if not state_input:
            self.conv_layers = [layers.Conv2D(16, 3, strides=3, activation='relu'), layers.Conv2D(64, 3, strides=2, activation='relu'), 
                                layers.Conv2D(128, 3, strides=2, activation='relu'), layers.Conv2D(256, 3, strides=2, activation='relu'), 
                                layers.GlobalAveragePooling2D()]
        else:
            self.conv_layers = []
            for unit in units:
                self.conv_layers.append(layers.Dense(unit, activation=hidden_activation))
        if self.residual:
            self.norm_layers = [layers.LayerNormalization()]* len(self.conv_layers)
        
        

        self.connect_layers = [layers.Dense(128, activation='relu'), layers.Dense(32, activation='relu')]
        if self.lstm:
            self.lstm_layers = [layers.LSTM(256,return_sequences=True),layers.GlobalAveragePooling1D()]+[layers.Dense(256, activation='relu')] 
            self.base_layers = self.conv_layers +self.lstm_layers + self.connect_layers
        else:
            self.base_layers = self.conv_layers + self.connect_layers

        self.out_mean = layers.Dense(action_dim, name="L_mean")
        if self._state_independent_std:
            self.out_logstd = tf.Variable(
                initial_value=-0.5 * np.ones(action_dim, dtype=np.float32),
                dtype=tf.float32, name="L_logstd")
        else:
            self.out_logstd = layers.Dense(action_dim, name="L_logstd")

        self._max_action = max_action

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        self(dummy_state)

    def _compute_dist(self, states):
        """

        Args:
            states: np.ndarray or tf.Tensor
                Inputs to neural network.

        Returns:
            tfp.distributions.MultivariateNormalDiag
                Multivariate normal distribution object whose mean and
                standard deviation is output of a neural network
        """
        features = states
        if self.residual:
            for i, cur_layer in enumerate(self.base_layers):
                if i<len(self.conv_layers):
                    if i==0:
                        features = self.norm_layers[i](cur_layer(features))
                    else:
                        features = features + self.norm_layers[i](cur_layer(features))
                else:
                    features = cur_layer(features)
        else:
            for cur_layer in self.base_layers:
                features = cur_layer(features)

        mean = self.out_mean(features)

        if self._state_independent_std:
            log_std = tf.tile(
                input=tf.expand_dims(self.out_logstd, axis=0),
                multiples=[mean.shape[0], 1])
        else:
            log_std = self.out_logstd(features)
            log_std = tf.clip_by_value(log_std, self.LOG_STD_CAP_MIN, self.LOG_STD_CAP_MAX)

        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=tf.exp(log_std))

    def call(self, states, test=False):
        """
        Compute actions and log probabilities of the selected action
        """
        dist = self._compute_dist(states)
        if test:
            raw_actions = dist.mean()
        else:
            raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)
        entropy = dist.entropy()

        if self._squash:
            actions = tf.tanh(raw_actions)
            diff = tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.EPS), axis=1)
            log_pis -= diff
        else:
            actions = raw_actions

        actions = actions * self._max_action
        return actions, log_pis,entropy

    def compute_log_probs(self, states, actions):
        raw_actions = actions / self._max_action
        dist = self._compute_dist(states)
        logp_pis = dist.log_prob(raw_actions)
        return logp_pis

    def compute_entropy(self, states):
        dist = self._compute_dist(states)
        return dist.entropy()
