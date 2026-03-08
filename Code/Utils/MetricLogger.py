from keras.callbacks import Callback
import logging

class MetricLogger(Callback):

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        msg = f"Epoch {epoch+1} | "

        for k, v in logs.items():
            msg += f"{k}: {v:.6f} | "

        logging.info(msg)