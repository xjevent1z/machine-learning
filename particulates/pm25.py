"""Get data and predict the PM value"""
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import tensorflow as tf


class PM25:
    """A class to predict PM 2.5 using linear regression."""
    train_x = list()
    train_y = list()
    test_x = list()
    pred_y_sk = list()
    pred_y_tf_adagard = list()
    pred_y_tf_sgd = list()
    pred_y_tf_adam = list()

    def read_train(self, file=str()):
        """Read a CSV file."""
        if os.path.exists(file) and file.endswith('csv'):
            with open(file, newline='', encoding='Big5') as csvfile:
                rows = csv.reader(csvfile, delimiter=',')
                for index, row in enumerate(rows):
                    if index != 0 and row[2] == 'PM2.5':
                        self.train_x.append([float(elem)
                                             for elem in row[3:12]])
                        self.train_y.append(float(row[13]))

    def read_test(self, file=str()):
        """Read a CSV file."""
        if os.path.exists(file) and file.endswith('csv'):
            with open(file, newline='', encoding='Big5') as csvfile:
                rows = csv.reader(csvfile, delimiter=',')
                for row in rows:
                    if row[1] == 'PM2.5':
                        self.test_x.append([float(elem)
                                            for elem in row[2:11]])

    def train_model_sklearn(self):
        """Train a model"""
        model = LinearRegression()
        model.fit(self.train_x, self.train_y)
        self.pred_y_sk = model.predict(self.test_x)

    def train_model_tensorflow(self):
        self.lr = 0.1
        self.x = tf.constant(self.train_x, dtype=tf.float32)
        self.y = tf.constant(self.train_y, dtype=tf.float32)

        self.adagard()
        self.sgd()
        self.adam()

    def adagard(self):
        w = tf.Variable([[0.1] for _ in range(9)], dtype=tf.float32)
        b = tf.Variable([np.zeros(shape=(1,))
                         for _ in range(240)], dtype=tf.float32)

        for _iter in range(100):
            optimizer = tf.optimizers.Adagrad(self.lr)
            with tf.GradientTape() as tape:
                y_hat = tf.add(tf.matmul(self.x, w), b)
                loss = tf.reduce_mean(tf.reduce_sum(tf.pow(self.y-y_hat, 2)))
            gradients = tape.gradient(loss, [w, b])
            optimizer.apply_gradients(zip(gradients, [w, b]))

        test_x = tf.constant(self.test_x, dtype=tf.float32)
        pred_y = tf.add(tf.matmul(test_x, w), b)
        pred_y = pred_y.numpy().T
        self.pred_y_tf_adagard = [elem for row in pred_y for elem in row]

    def sgd(self):
        w = tf.Variable([[0.1] for _ in range(9)], dtype=tf.float32)
        b = tf.Variable([np.zeros(shape=(1,))
                         for _ in range(240)], dtype=tf.float32)
        for _iter in range(100):
            optimizer = tf.optimizers.SGD(self.lr)
            with tf.GradientTape() as tape:
                y_hat = tf.add(tf.matmul(self.x, w), b)
                loss = tf.reduce_mean(tf.reduce_sum(tf.pow(self.y-y_hat, 2)))
            gradients = tape.gradient(loss, [w, b])
            optimizer.apply_gradients(zip(gradients, [w, b]))

        test_x = tf.constant(self.test_x, dtype=tf.float32)
        pred_y = tf.add(tf.matmul(test_x, w), b)
        pred_y = pred_y.numpy().T
        self.pred_y_tf_sgd = [elem for row in pred_y for elem in row]

    def adam(self):
        w = tf.Variable([[0.1] for _ in range(9)], dtype=tf.float32)
        b = tf.Variable([np.zeros(shape=(1,))
                         for _ in range(240)], dtype=tf.float32)
        for _iter in range(100):
            optimizer = tf.optimizers.Adam(self.lr)
            with tf.GradientTape() as tape:
                y_hat = tf.add(tf.matmul(self.x, w), b)
                loss = tf.reduce_mean(tf.reduce_sum(tf.pow(self.y-y_hat, 2)))
            gradients = tape.gradient(loss, [w, b])
            optimizer.apply_gradients(zip(gradients, [w, b]))

        test_x = tf.constant(self.test_x, dtype=tf.float32)
        pred_y = tf.add(tf.matmul(test_x, w), b)
        pred_y = pred_y.numpy().T
        self.pred_y_tf_adam = [elem for row in pred_y for elem in row]

    def draw(self):
        """Draw the figure"""
        plt.plot(np.arange(len(self.pred_y_sk)), np.array(
            self.pred_y_sk), 'r--', label='sk')
        plt.plot(np.arange(len(self.pred_y_tf_adagard)), np.array(
            self.pred_y_tf_adagard), 'g--', label='tf_adagrad')
        # plt.plot(np.arange(len(self.pred_y_tf_sgd)), np.array(
        #     self.pred_y_tf_sgd), 'b--', label='tf_sgd')
        # plt.plot(np.arange(len(self.pred_y_tf_adam)), np.array(
        #     self.pred_y_tf_adam), 'r--', label='tf_adam')
        plt.title('PM 2.5 prediction')
        plt.legend()
        plt.show()
