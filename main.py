"""The main program to call"""
from particulates import PM25
import tensorflow as tf
import numpy as np


def test():
    X = np.random.rand(10).astype(np.float32)
    Y = X * 10 + 5
    W = tf.Variable(tf.random.normal([1]))
    b = tf.Variable(tf.zeros([1]))
    optimizer = tf.optimizers.Adagrad(0.1)
    for step in range(1, 1000 + 1):
        # Run the optimization to update W and b values.
        with tf.GradientTape() as tape:
            y_hat = W * X + b
            loss = tf.reduce_sum(tf.pow(y_hat-Y, 2)) / (2 * 10)
        gradients = tape.gradient(loss, [W, b])
        optimizer.apply_gradients(zip(gradients, [W, b]))
        if step % 10 == 0:
            print("step: %i, loss: %f, W: %f, b: %f" %
                  (step, loss, W.numpy(), b.numpy()))


if __name__ == "__main__":
    PM = PM25()
    PM.read_train('particulates/input/train.csv')
    PM.read_test('particulates/input/test.csv')
    PM.train_model_tensorflow()
    PM.train_model_sklearn()
    PM.draw()
