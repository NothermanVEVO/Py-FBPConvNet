import tensorflow as tf

def psnr_metric(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mse = tf.maximum(mse, 1e-8)
    return -10.0 * tf.math.log(mse) / tf.math.log(10.0)