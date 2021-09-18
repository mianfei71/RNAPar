from keras.callbacks import Callback
import keras.backend as K

class StepLR(Callback):

    def __init__(self, lr_list, epoch_list, verbose=1, info=""):
        super().__init__()

        self.lr_list = lr_list
        self.epoch_list = epoch_list
        self.verbose = verbose
        self.info = info

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.epoch_list[0]:
            K.set_value(self.model.optimizer.lr, self.lr_list[0])
        elif epoch == self.epoch_list[0]:
            self.lr_list.pop(0)
            self.epoch_list.pop(0)
            K.set_value(self.model.optimizer.lr, self.lr_list[0])
        if self.verbose > 0:
            print('\nEpoch %05d: Learning Rate is %s, %s.' % (epoch + 1, K.get_value(self.model.optimizer.lr), self.info))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

