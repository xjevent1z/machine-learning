"""The main program to call"""
# from particulates import PM25
# from weather import Forcast
from classification import Income

# def test():
#     X = np.random.rand(10).astype(np.float32)
#     Y = X * 10 + 5
#     W = tf.Variable(tf.random.normal([1]))
#     b = tf.Variable(tf.zeros([1]))
#     optimizer = tf.optimizers.Adagrad(0.1)
#     for step in range(1, 1000 + 1):
#         # Run the optimization to update W and b values.
#         with tf.GradientTape() as tape:
#             y_hat = W * X + b
#             loss = tf.reduce_sum(tf.pow(y_hat-Y, 2)) / (2 * 10)
#         gradients = tape.gradient(loss, [W, b])
#         optimizer.apply_gradients(zip(gradients, [W, b]))
#         if step % 10 == 0:
#             print("step: %i, loss: %f, W: %f, b: %f" %
#                   (step, loss, W.numpy(), b.numpy()))


# def f(x):
#     return x ** 2 * 2 + x * 3 + 4


# def test():
#     x = tf.constant([[2]], dtype=float)
#     y = f(x)
#     w1 = tf.Variable([np.random.rand(1)], dtype=float)
#     w2 = tf.Variable([np.random.rand(1)], dtype=float)
#     b = tf.Variable([np.random.rand(1)], dtype=float)

#     optimizer = tf.optimizers.Adagrad(0.1)
#     for _iter in range(10000):
#         with tf.GradientTape() as tape:
#             y_hat = tf.add_n(
#                 [tf.matmul(tf.pow(x, 2), w1), tf.matmul(x, w2), b])
#             loss = tf.reduce_mean(tf.reduce_sum(tf.pow(y - y_hat, 2)))

#         gradients = tape.gradient(loss, [w1, w2, b])
#         optimizer.apply_gradients(zip(gradients, [w1, w2, b]))

#     m = tf.constant([[7]], dtype=float)
#     y = f(m)
#     y_pred = tf.add_n(
#         [tf.matmul(tf.pow(m, 2), w1), tf.matmul(m, w2), b])
#     print("y_real = {}, y_pred = {}".format(
#         float(y.numpy()), float(y_pred.numpy())))


if __name__ == "__main__":
    # PM = PM25()
    # PM.read_train('particulates/input/train.csv')
    # PM.read_test('particulates/input/test.csv')
    # PM.train_model_tensorflow()
    # PM.train_model_sklearn()
    # PM.draw()
    # agent = Forcast()
    # agent.read_train()
    # agent.read_test()
    # agent.train_model()
    # test()
    AGENT = Income()
    AGENT.encode('classification/simple.csv')
    AGENT.training()
