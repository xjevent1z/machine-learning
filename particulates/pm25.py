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
    pred_y = list()

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
        self.pred_y = model.predict(self.test_x)

    # def train_model_tensorflow(self):
    #     lr = 0.00001
    #     x = tf.constant(self.train_x, dtype=tf.float32)
    #     y = tf.constant(self.train_y, dtype=tf.float32)
    #     w = tf.Variable([[0.1] for _ in range(9)], dtype=tf.float32)
    #     b = tf.Variable([np.zeros(shape=(1,))
    #                      for _ in range(240)], dtype=tf.float32)
    #     y_hat = tf.add(tf.matmul(x, w), b)
    #     loss = tf.reduce_mean(tf.reduce_sum(tf.pow(y-y_hat, 2)))

    #     optm_w, optm_b = self.tensor_GD(w[0], b[0], lr, loss, 0)

    # def tensor_GD(self, w, b, lr, loss, _iter):
    #     if _iter == '1000':
    #         return w, b
    #     else:
    #         grad_w, grad_b = tf.gradients(loss, [w, b])
    #         new_w = tf.assign(w, w - lr * grad_w)
    #         new_b = tf.assign(b, b - lr * grad_b)
    #         _iter += 1
    #         return self.tensor_GD(new_w, new_b, _iter)

    def draw(self):
        """Draw the figure"""
        plt.plot(np.arange(len(self.pred_y)), np.array(
            self.pred_y), 'r--', label='sk_learn')
        plt.legend()
        plt.show()
