import tensorflow as tf

class PSNRLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        mse = tf.maximum(mse, 1e-8)
        psnr = -10.0 * tf.math.log(mse) / tf.math.log(10.0)
        return -psnr

class MSE_SSIM_Loss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        ssim = tf.reduce_mean(
            tf.image.ssim(y_true, y_pred, max_val=1.0)
        )

        loss = self.alpha * mse + self.beta * (1.0 - ssim)

        return loss

class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        ssim = tf.image.ssim(
            y_true,
            y_pred,
            max_val=1.0
        )
        
        ssim_mean = tf.reduce_mean(ssim)

        return 1.0 - ssim_mean