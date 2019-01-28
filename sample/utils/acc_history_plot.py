import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from keras.callbacks import Callback

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
        self.train_accs.append(logs.get('acc'))
        self.test_loss.append(test_loss)
        self.train_loss.append(logs.get('loss'))
        if self.validate:
            self.val_accs.append(logs.get('val_acc'))
            self.val_loss.append(logs.get('val_loss'))
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