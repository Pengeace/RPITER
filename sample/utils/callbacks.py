import matplotlib
matplotlib.use("Agg")

import warnings
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback


# to consider version compatibility
def get_monitor_value(logs, monitor):
    if (monitor == 'acc') and ('acc' not in logs.keys()):
        monitor = 'accuracy'
    monitor_value = logs.get(monitor)
    if monitor_value is None:
        warnings.warn(
            'Early stopping conditioned on metric `%s` '
            'which is not available. Available metrics are: %s' %
            (monitor, ','.join(list(logs.keys()))), RuntimeWarning
        )
    return monitor_value


class AccHistoryPlot(Callback):
    def __init__(self, stage_infos, test_data, data_name, result_save_path, validate=0, plot_epoch_gap=30, verbose=1):
        super(AccHistoryPlot, self).__init__()
        self.stage, self.fold = stage_infos
        self.X_test, self.y_test = test_data
        self.data_name = data_name
        self.result_save_path = result_save_path
        self.plot_epoch_gap = plot_epoch_gap
        self.validate = validate
        self.verbose = verbose

        self.close_plt_on_train_end = True

        fig = plt.figure()
        self.ax = fig.add_subplot(1, 1, 1)
        plt.xlabel('Training epochs', fontsize=13)
        plt.ylabel('Accuracy and loss values', fontsize=13)
        plt.title('$Acc$ and loss of {} on {}'.format(self.stage, self.data_name), fontsize=15)
        plt.ion()

        self.train_accs = []
        self.train_loss = []
        self.val_accs = []
        self.val_loss = []
        self.test_accs = []
        self.test_loss = []

        self.line_train_acc = None
        self.line_val_acc = None
        self.line_test_acc = None
        self.line_train_loss = None
        self.line_val_loss = None
        self.line_test_loss = None

    def plot(self):
        if self.line_train_acc:
            self.ax.lines.remove(self.line_train_acc[0])
            self.ax.lines.remove(self.line_test_acc[0])
            self.ax.lines.remove(self.line_train_loss[0])
            self.ax.lines.remove(self.line_test_loss[0])
            if self.validate:
                self.ax.lines.remove(self.line_val_acc[0])
                self.ax.lines.remove(self.line_val_acc[0])
        self.line_train_acc = self.ax.plot(self.train_accs, lw=1.8, color='deepskyblue', label='Train $Acc$')
        if self.validate:
            self.line_val_acc = self.ax.plot(self.val_accs, lw=1.8, color='gold', label='Val $Acc$')
        self.line_test_acc = self.ax.plot(self.test_accs, lw=1.8, color='limegreen', label = 'Test $Acc$')
        self.line_train_loss = self.ax.plot(self.train_loss, lw=1.8, color='coral', label='Train Loss')
        if self.validate:
            self.line_val_acc = self.ax.plot(self.val_accs, lw=1.8, color='darkred', label='Val Loss')
        self.line_test_loss = self.ax.plot(self.test_loss, lw=1.8, color='darkorange', label = 'Test Loss')


        self.ax.legend(loc='center right' if not self.validate else 'best', fontsize=10)
        plt.pause(0.1)

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_acc = self.model.evaluate(x=self.X_test, y=self.y_test, batch_size=200, verbose=0)
        if self.verbose:
            print("Current loss: {}, acc: {}".format(test_loss, test_acc))
        self.test_accs.append(test_acc)
        self.train_accs.append(get_monitor_value(logs, 'acc'))
        self.test_loss.append(test_loss)
        self.train_loss.append(get_monitor_value(logs, 'loss'))
        if self.validate:
            self.val_accs.append(get_monitor_value(logs, 'val_acc'))
            self.val_loss.append(get_monitor_value(logs, 'val_loss'))
        if epoch % self.plot_epoch_gap == 0:
            self.plot()

    def on_train_end(self, logs=None):
        self.plot()

        if self.fold!=None:
            plot_file_name = 'fold {} of {} stage.pdf'.format(self.fold, self.stage)
        else:
            plot_file_name = '{} stage.pdf'.format(self.stage)

        plt.savefig(self.result_save_path + plot_file_name)
        if self.close_plt_on_train_end:
            plt.close()


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs that produced the monitored
            quantity with no improvement after which training will
            be stopped.
            Validation quantities may not be produced for every
            epoch, if the validation frequency
            (`model.fit(validation_freq=5)`) is greater than one.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = get_monitor_value(logs, self.monitor)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
