"""Get data and predict the PM value"""
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class Forcast:
    """A class to predict PM 2.5 using linear regression."""
    train_x = list()
    train_y = list()
    test_x = list()
    real_y = list()
    pred_y = list()

    def read_train(self):
        """Generate training data set"""
        x = list()
        y = list()
        for i in range(1, 13):
            _file = os.path.join(os.path.dirname(__file__), 'input', '466920-2018-{:02d}.csv'.format(i))
            with open(_file, newline='', encoding='utf-8-sig') as csvfile:
                rows = csv.reader(csvfile)
                x.append([float(row[7]) for index, row in enumerate(rows) if index in range(2, 22)])

            with open(_file, newline='', encoding='utf-8-sig') as csvfile:
                rows = csv.reader(csvfile)
                data = [float(row[7]) for index, row in enumerate(rows) if index > 2]
                y.append(np.average(data))

        self.train_x = tf.constant(x, dtype=float)  # 12 x 20
        self.train_y = tf.constant(y, dtype=float)  # 1 x 12

    def read_test(self):
        """Generate testing data set"""
        x = list()
        y = list()
        for i in range(1, 13):
            _file = os.path.join(os.path.dirname(__file__), 'input', '466920-2019-{:02d}.csv'.format(i))
            with open(_file, newline='', encoding='utf-8-sig') as csvfile:
                rows = csv.reader(csvfile)
                x.append([float(row[7]) for index, row in enumerate(rows) if index in range(2, 22)])

            with open(_file, newline='', encoding='utf-8-sig') as csvfile:
                rows = csv.reader(csvfile)
                data = [float(row[7]) for index, row in enumerate(rows) if index > 2]
                y.append(np.average(data))
        self.test_x = tf.constant(x, dtype=float)  # 1 x 12
        self.real_y = tf.constant(y, dtype=float)  # 1 x 12

    def train_model(self):
        """Train a linear model using linear regression"""
        _rand = float(np.random.rand(1))
        _list = [[_rand] for _ in range(len(self.train_x[0]))]
        w1 = tf.Variable(_list, dtype=float)  # 20 x 1

        _rand = float(np.random.rand(1))
        _list = [[_rand] for _ in range(len(self.train_x[0]))]
        w2 = tf.Variable(_list, dtype=float)  # 20 x 1

        _rand = float(np.random.rand(1))
        _list = [[_rand] for _ in range(len(self.train_x))]
        b = tf.Variable(_list, dtype=float)  # 12 x 1

        optimizer = tf.optimizers.SGD(1e-7)
        _lambda = 1e-2
        for _iter in range(10000):
            with tf.GradientTape() as tape:
                y_hat = tf.add_n([tf.matmul(tf.pow(self.train_x, 2), w1), tf.matmul(self.train_x, w2), b])
                loss = tf.reduce_mean(tf.pow(y_hat - self.train_y, 2)) + \
                    _lambda * tf.reduce_sum(tf.pow(w1, 2)) / (tf.reduce_sum(tf.pow(w1, 2)) + tf.reduce_sum(tf.pow(w2, 2))) + \
                    _lambda * tf.reduce_sum(tf.pow(w2, 2)) / (tf.reduce_sum(tf.pow(w1, 2)) + tf.reduce_sum(tf.pow(w2, 2)))
            gradients = tape.gradient(loss, [w1, w2, b])
            optimizer.apply_gradients(zip(gradients, [w1, w2, b]))
        self.y_pred1 = tf.add_n([tf.matmul(tf.pow(self.train_x, 2), w1), tf.matmul(self.train_x, w2), b])
        self.y_pred2 = tf.add_n([tf.matmul(tf.pow(self.test_x, 2), w1), tf.matmul(self.test_x, w2), b])

        # _range = [float('1e{}'.format(exp)) for exp in np.linspace(-3,2,6, dtype=int)]
        # for index, _lambda in enumerate(_range):
        #     for _iter in range(10000):
        #         with tf.GradientTape() as tape:
        #             y_hat = tf.add_n([tf.matmul(tf.pow(self.train_x, 2), w1), tf.matmul(self.train_x, w2), b])
        #             loss = tf.reduce_mean(tf.pow(y_hat - self.train_y, 2)) + \
        #                 _lambda * tf.reduce_sum(tf.pow(w1, 2)) / (tf.reduce_sum(tf.pow(w1, 2)) + tf.reduce_sum(tf.pow(w2, 2))) + \
        #                 _lambda * tf.reduce_sum(tf.pow(w2, 2)) / (tf.reduce_sum(tf.pow(w1, 2)) + tf.reduce_sum(tf.pow(w2, 2)))
        #         gradients = tape.gradient(loss, [w1, w2, b])
        #         optimizer.apply_gradients(zip(gradients, [w1, w2, b]))

        #     self.y_pred1 = tf.add_n([tf.matmul(tf.pow(self.train_x, 2), w1), tf.matmul(self.train_x, w2), b])
        #     self.y_pred2 = tf.add_n([tf.matmul(tf.pow(self.test_x, 2), w1), tf.matmul(self.test_x, w2), b])

        #     if index == 0:
        #         plt.plot(np.arange(1, 13), self.y_pred1, 'r:', label='lambda_{}'.format(_lambda))
        #     if index == 1:
        #         plt.plot(np.arange(1, 13), self.y_pred1, 'g:', label='lambda_{}'.format(_lambda))
        #     if index == 2:
        #         plt.plot(np.arange(1, 13), self.y_pred1, 'b:', label='lambda_{}'.format(_lambda))
        #     if index == 3:
        #         plt.plot(np.arange(1, 13), self.y_pred1, 'c:', label='lambda_{}'.format(_lambda))
        #     if index == 4:
        #         plt.plot(np.arange(1, 13), self.y_pred1, 'y:', label='lambda_{}'.format(_lambda))
        #     if index == 5:
        #         plt.plot(np.arange(1, 13), self.y_pred1, 'm:', label='lambda_{}'.format(_lambda))

        #     loss = tf.reduce_mean(tf.pow(self.y_pred1 - self.train_y, 2))
        #     print('Loss = {} with lambda {}'.format(loss, _lambda))

        # plt.xticks(range(1, 13))
        # plt.legend()
        # plt.savefig(os.path.join(os.path.dirname(__file__), 'lambda.png'))
        # plt.close()

    def draw(self, tag, _file):
        """Draw the figure"""
        if tag == 'train':
            plt.plot(np.arange(1, 13), self.train_y, 'r.', label='y_train')
            plt.plot(np.arange(1, 13), self.y_pred1, 'm:', label='y_pred')
            loss = tf.reduce_mean(tf.pow(self.y_pred1 - self.train_y, 2))
        elif tag == 'test':
            plt.plot(np.arange(1, 13), self.real_y, 'bD', label='y_real')
            plt.plot(np.arange(1, 13), self.y_pred2, 'm:', label='y_pred')
            loss = tf.reduce_mean(tf.pow(self.y_pred2 - self.real_y, 2))
        plt.xticks(range(1, 13))
        plt.title('Loss = {}'.format(loss))
        plt.legend()
        plt.savefig(os.path.join(os.path.dirname(__file__), _file))
        plt.close()
