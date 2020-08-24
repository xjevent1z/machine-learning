"""Get data and predict the PM value"""
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf


class Forcast:
    """A class to predict PM 2.5 using linear regression."""
    train_x = list()
    train_y = list()
    test_x = list()
    pred_y = list()

    def read_train(self):
        """Generate training data set"""
        x = list()
        y = list()
        for i in range(1, 13):
            with open(os.path.join('weather', 'input', '466920-2018-{:02d}.csv'.format(i)), 'r') as fh:
                rows = csv.reader(fh)

                x.append([float(row[7]) for index, row in enumerate(rows)
                          if index not in (0, 1) and index <= 21])
        self.train_x = tf.constant(x, dtype=float)  # 12 x 20 matrix

        for i in range(1, 13):
            with open(os.path.join('weather', 'input', '466920-2018-{:02d}.csv'.format(i)), 'r') as fh:
                rows = csv.reader(fh)

                y.append([float(row[7]) for index, row in enumerate(rows)
                          if index not in (0, 1) and index == 22])

        self.train_y = tf.constant(y, dtype=float)  # 12 x 1 matrix

    def read_test(self):
        """Generate testing data set"""
        x = list()
        y = list()
        for i in range(1, 13):
            with open(os.path.join('weather', 'input', '466920-2019-{:02d}.csv'.format(i)), 'r') as fh:
                rows = csv.reader(fh)

                x.append([float(row[7]) for index, row in enumerate(rows)
                          if index not in (0, 1) and index <= 21])
        self.test_x = tf.constant(x, dtype=float)  # 12 x 20 matrix

        for i in range(1, 13):
            with open(os.path.join('weather', 'input', '466920-2019-{:02d}.csv'.format(i)), 'r') as fh:
                rows = csv.reader(fh)

                y.append([float(row[7]) for index, row in enumerate(rows)
                          if index not in (0, 1) and index == 22])

        self.test_y = tf.constant(y, dtype=float)  # 12 x 1 matrix

    def train_model(self):
        """Train a linear model using linear regression"""
        # _w = tf.Variable(np.random.rand(1), dtype=float)
        # _b = tf.Variable(np.random.rand(1), dtype=float)

        _rand = np.random.rand(1)
        _list = [_rand for _ in range(len(self.train_x[0]))]
        _w1 = tf.Variable(_list, dtype=float)  # 20 x 1 matrix
        _rand = np.random.rand(1)
        _list = [_rand for _ in range(len(self.train_x[0]))]
        _w2 = tf.Variable(_list, dtype=float)  # 20 x 1 matrix
        _rand = np.random.rand(1)
        _list = [_rand for _ in range(len(self.train_x))]
        _b = tf.Variable(_list, dtype=float)  # 12 x 1 matrix

        # for index, lr in enumerate(np.linspace(0.01, 1.00, 11)):
        #     optimizer = tf.optimizers.Adagrad(lr)
        #     w1, w2, b = _w1, _w2, _b
        #     for _iter in range(1000):
        #         with tf.GradientTape() as tape:
        #             y_hat = tf.add_n([tf.matmul(tf.pow(self.train_x, 2), w1), tf.matmul(
        #                 self.train_x, w2), b])  # 12 x 20 matrix
        #             loss = tf.reduce_mean(tf.pow(self.train_y - y_hat, 2))
        #         gradients = tape.gradient(loss, [w1, w2, b])
        #         optimizer.apply_gradients(zip(gradients, [w1, w2, b]))
        #     y_pred = tf.add_n([tf.matmul(tf.pow(self.train_x, 2), w1), tf.matmul(
        #         self.train_x, w2), b])

        #     if index == 0:
        #         plt.plot(np.arange(len(self.train_x)),
        #                  y_pred, 'r-', label='lr{:.2f}'.format(lr))
        #     if index == 1:
        #         plt.plot(np.arange(len(self.train_x)),
        #                  y_pred, 'r--', label='lr{:.2f}'.format(lr))
        #     if index == 2:
        #         plt.plot(np.arange(len(self.train_x)),
        #                  y_pred, 'r-.', label='lr{:.2f}'.format(lr))
        #     if index == 3:
        #         plt.plot(np.arange(len(self.train_x)),
        #                  y_pred, 'r:', label='lr{:.2f}'.format(lr))
        #     if index == 4:
        #         plt.plot(np.arange(len(self.train_x)),
        #                  y_pred, 'g-', label='lr{:.2f}'.format(lr))
        #     if index == 5:
        #         plt.plot(np.arange(len(self.train_x)),
        #                  y_pred, 'g--', label='lr{:.2f}'.format(lr))
        #     if index == 6:
        #         plt.plot(np.arange(len(self.train_x)),
        #                  y_pred, 'g-.', label='lr{:.2f}'.format(lr))
        #     if index == 7:
        #         plt.plot(np.arange(len(self.train_x)),
        #                  y_pred, 'g:', label='lr{:.2f}'.format(lr))
        #     if index == 8:
        #         plt.plot(np.arange(len(self.train_x)),
        #                  y_pred, 'b-', label='lr{:.2f}'.format(lr))
        #     if index == 9:
        #         plt.plot(np.arange(len(self.train_x)),
        #                  y_pred, 'b--', label='lr{:.2f}'.format(lr))
        #     if index == 10:
        #         plt.plot(np.arange(len(self.train_x)),
        #                  y_pred, 'b-.', label='lr{:.2f}'.format(lr))
        #     if index == 11:
        #         plt.plot(np.arange(len(self.train_x)),
        #                  y_pred, 'b:', label='lr{:.2f}'.format(lr))
        # plt.legend()
        # plt.xticks([i for i in range(0, 13)])
        # plt.show()
        optimizer = tf.optimizers.Adagrad(1.01)
        w1, w2, b = _w1, _w2, _b
        for _iter in range(10000):
            with tf.GradientTape() as tape:
                y_hat = tf.add_n([tf.matmul(tf.pow(self.train_x, 2), w1), tf.matmul(
                    self.train_x, w2), b])  # 12 x 20 matrix
                loss = tf.reduce_mean(tf.pow(self.train_y - y_hat, 2))
            gradients = tape.gradient(loss, [w1, w2, b])
            optimizer.apply_gradients(zip(gradients, [w1, w2, b]))
        y_pred = tf.add_n([tf.matmul(tf.pow(self.train_x, 2), w1), tf.matmul(
            self.train_x, w2), b])

        plt.plot(np.arange(len(self.train_x)),
                 self.train_y, 'y^:', label='y_train')
        plt.plot(np.arange(len(self.train_x)),
                 y_pred, 'r--', label='y_pred')
        plt.legend()
        plt.show()

    def draw(self):
        """Draw the figure"""
        plt.plot(np.arange(len(self.pred_y_sk)), np.array(
            self.pred_y_sk), 'r--', label='sk')
        plt.plot(np.arange(len(self.pred_y_tf_adagard)), np.array(
            self.pred_y_tf_adagard), 'g--', label='tf_adagrad')
        plt.title('Weather Forecast')
        plt.legend()
        plt.show()
