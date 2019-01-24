"""

"""

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import adam

from src.Datagen import DataSequence
from src.backend import ENET, VGG, UNET
from keras import callbacks

from keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat),
                        axis=-1)

class Segment(object):

    def __init__(self, backend,
                 input_size, nb_classes):

        """
        Model Factory that fetches the corresponding model based on the backend that has been defined
        and initiates the training process

        :param backend: define the backbone architecture for the training
        :param input_size: the size of the input image
        :param nb_classes: the number of classes
        """
        self.input_size = input_size
        self.nb_classes = nb_classes

        if backend == "ENET":
            self.feature_extractor = ENET(self.input_size, self.nb_classes).build()
        elif backend == "VGG":
            self.feature_extractor = VGG(self.input_size, self.nb_classes).build()
        elif backend == "UNET":
            self.feature_extractor = UNET(self.input_size, self.nb_classes).build()
        else:
            raise ValueError('No such arch!... Please check the backend in config file')

    def train(self, train_configs):

        """
         Train the model based on the training configurations
        :param train_configs: Configuration for the training
        """
        optimizer = adam(train_configs["learning_rate"])
        train_times = train_configs["train_times"]

        # Data sequence for training
        sequence = DataSequence(train_configs["data_directory"], train_configs["batch_size"],
                                self.input_size)
        steps_per_epoch = len(sequence) * train_times

        # configure the model for training

        # self.feature_extractor.compile(optimizer=optimizer, loss="categorical_crossentropy",
        #                                metrics=['accuracy'])
        self.feature_extractor.compile(optimizer=optimizer, loss=dice_coef_loss,
                                       metrics=['accuracy'])
        # define the callbacks for training
        tb = TensorBoard(log_dir=train_configs["logs_dir"], write_graph=True)
        mc = ModelCheckpoint(mode='max', filepath=train_configs["save_model_name"], monitor='acc',
                             save_best_only='True',
                             save_weights_only='True', verbose=2)
        es = EarlyStopping(mode='max', monitor='acc', patience=6, verbose=1)
        model_reducelr = callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.2,
            patience=5,
            verbose=1,
            min_lr=0.05 * train_configs["learning_rate"])

        callback = [tb, mc, es, model_reducelr]

        # Train the model on data generated batch-by-batch by the DataSequence generator
        self.feature_extractor.fit_generator(sequence,
                                             steps_per_epoch=steps_per_epoch,
                                             epochs=train_configs["nb_epochs"],
                                             verbose=1,
                                             shuffle=True, callbacks=callback,
                                             workers=3,
                                             max_queue_size=8,
                                             class_weight="auto"
                                             )
