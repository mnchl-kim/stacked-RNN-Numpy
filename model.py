import numpy as np
from utils import softmax, cross_entropy, cross_entropy_d, initialize_xavier


class RNN:
    def __init__(self, input_size, output_size, hidden_size, cell_length, depth_size=1, batch_size=1, drop_rate=0):
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self._cell_length = cell_length
        self._depth_size = depth_size
        self._batch_size = batch_size
        self._drop_rate = drop_rate

        # Xavier initialization
        self._parameters = {'W_xa': [initialize_xavier(self._input_size if d == 0 else self._hidden_size, self._hidden_size) for d in range(self._depth_size)],
                            'W_aa': [initialize_xavier(self._hidden_size, self._hidden_size) for d in range(self._depth_size)],
                            'W_ay': [initialize_xavier(self._hidden_size, self._output_size)],
                            'b_a': [np.zeros((1, self._hidden_size)) for d in range(self._depth_size)],
                            'b_y': [np.zeros((1, self._output_size))],
                            'a': [np.zeros((self._batch_size, self._hidden_size)) for d in range(self._depth_size)]}

        self._gradients = {'dW_xa': [np.zeros_like(self._parameters['W_xa'][d]) for d in range(self._depth_size)],
                           'dW_aa': [np.zeros_like(self._parameters['W_aa'][d]) for d in range(self._depth_size)],
                           'dW_ay': [np.zeros_like(self._parameters['W_ay'][0])],
                           'db_a': [np.zeros_like(self._parameters['b_a'][d]) for d in range(self._depth_size)],
                           'db_y': [np.zeros_like(self._parameters['b_y'][0])],
                           'da': [np.zeros_like(self._parameters['a'][d]) for d in range(self._depth_size)]}

        self._momentums = {'dW_xa': [np.ones_like(self._gradients['dW_xa'][d]) * 0.1 for d in range(self._depth_size)],
                           'dW_aa': [np.zeros_like(self._gradients['dW_aa'][d]) * 0.1 for d in range(self._depth_size)],
                           'dW_ay': [np.zeros_like(self._gradients['dW_ay'][0]) * 0.1],
                           'db_a': [np.zeros_like(self._gradients['db_a'][d]) * 0.1 for d in range(self._depth_size)],
                           'db_y': [np.zeros_like(self._gradients['db_y'][0]) * 0.1]}

        self._loss = - np.log(1.0 / self._output_size) * self._cell_length

    def optimize(self, X, Y, learning_rate=0.01):
        cache = self.forward(X)
        self.backward(Y, cache)
        self.update_parameters(learning_rate=learning_rate)

    def forward(self, X):
        self._loss = 0

        x, y_hat = [{} for d in range(self._depth_size + 1)], {}
        a = [{-1: np.copy(self._parameters['a'][d])} for d in range(self._depth_size)]
        dropout = [{} for d in range(self._depth_size)]

        for t in range(self._cell_length):
            x[0][t] = X[:, t, :, :].reshape(self._batch_size, self._input_size)

            for d in range(self._depth_size):
                dropout[d][t] = np.random.binomial(1, 1 - self._drop_rate, (1, self._hidden_size)) / (1 - self._drop_rate)
                a[d][t] = np.tanh(np.dot(x[d][t], self._parameters['W_xa'][d]) +
                                  np.dot(a[d][t - 1], self._parameters['W_aa'][d]) +
                                  self._parameters['b_a'][d])
                x[d + 1][t] = np.copy(a[d][t]) * dropout[d][t]

            z = np.dot(x[self._depth_size][t], self._parameters['W_ay'][0]) + self._parameters['b_y'][0]
            z = np.clip(z, -100, 100)
            y_hat[t] = np.array([softmax(z[b, :]) for b in range(self._batch_size)])

        cache = (x, a, y_hat, dropout)
        return cache

    def backward(self, Y, cache):
        self._gradients = {key: [np.zeros_like(self._gradients[key][d]) for d in range(len(self._gradients[key]))] for key in self._gradients.keys()}
        (x, a, y_hat, dropout) = cache

        for t in reversed(range(self._cell_length)):
            self._loss += sum([cross_entropy(y_hat[t][b, :], Y[b, t]) for b in range(self._batch_size)]) / (self._cell_length * self._batch_size)
            dy = np.array([cross_entropy_d(y_hat[t][b, :], Y[b, t]) for b in range(self._batch_size)]) / (self._cell_length * self._batch_size)

            self._gradients['dW_ay'][0] += np.dot(x[self._depth_size][t].T, dy)
            self._gradients['db_y'][0] += dy.sum(axis=0)
            da = np.dot(dy, self._parameters['W_ay'][0].T)

            for d in reversed(range(self._depth_size)):
                da = (1 - a[d][t] ** 2) * (da * dropout[d][t] + self._gradients['da'][d])
                self._gradients['dW_xa'][d] += np.dot(x[d][t].T, da)
                self._gradients['dW_aa'][d] += np.dot(a[d][t - 1].T, da)
                self._gradients['db_a'][d] += da.sum(axis=0)
                self._gradients['da'][d] = np.dot(da, self._parameters['W_aa'][d].T)
                da = np.dot(da, self._parameters['W_xa'][d].T)

        self._parameters['a'] = [a[d][self._cell_length - 1] for d in range(self._depth_size)]

    def update_parameters(self, learning_rate=0.01):
        parameters = self._parameters['W_xa'] + self._parameters['W_aa'] + self._parameters['W_ay'] + self._parameters['b_a'] + self._parameters['b_y']
        gradients = self._gradients['dW_xa'] + self._gradients['dW_aa'] + self._gradients['dW_ay'] + self._gradients['db_a'] + self._gradients['db_y']
        momentums = self._momentums['dW_xa'] + self._momentums['dW_aa'] + self._momentums['dW_ay'] + self._momentums['db_a'] + self._momentums['db_y']

        for w, g, m in zip(parameters, gradients, momentums):
            np.clip(w, -1, 1, out=w)

            # # Adagrad
            # m += g ** 2
            # w -= learning_rate * g / np.sqrt(m + 1e-8)

            # RMSProp
            m = 0.9 * m + 0.1 * g ** 2
            w -= learning_rate * g / np.sqrt(m + 1e-8)

    def sample(self, ix, n):
        ixes = [ix]
        a = [np.zeros((1, self._hidden_size)) for d in range(self._depth_size)]
        for t in range(n):
            x = np.zeros((1, self._input_size))
            x[0, ix] = 1

            for d in range(self._depth_size):
                a[d] = np.tanh(np.dot(x, self._parameters['W_xa'][d]) +
                               np.dot(a[d], self._parameters['W_aa'][d]) +
                               self._parameters['b_a'][d])
                x = a[d]

            z = np.dot(x, self._parameters['W_ay']) + self._parameters['b_y']
            z = np.clip(z, -100, 100)
            y = softmax(z / 0.7)

            ix = np.random.choice(range(self._input_size), p=y.ravel())
            ixes.append(ix)

        return ixes

    def initialize_optimizer(self):
        self._momentums = {key: [np.ones_like(self._momentums[key][d]) * 0.1 for d in range(len(self._momentums[key]))] for key in self._momentums.keys()}

    def initialize_hidden_state(self):
        self._parameters['a'] = [np.zeros_like(self._parameters['a'][d]) for d in range(self._depth_size)]

    def hidden_state(self):
        return self._parameters['a']

    def loss(self):
        return self._loss

    def parameters(self):
        return self._parameters
