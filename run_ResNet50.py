import numpy as np
from glob import glob
from scipy import ndimage
from keras import callbacks
from keras.optimizers import Adamax, SGD, RMSprop

import resnet50

def convert_to_one_hot(Y, C):
    '''Converts array with labels to one-hot encoding
    
    Keyword Arguments:
    Y -- 1-dimensional numpy array containing labeled values
    C -- total number of labels in Y
    '''

    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def load_dataset(datapath, composers):
    '''Loads dataset into memory

    Keyword Arguments:
    datapath -- absolute or relative path to dataset location
    composers -- list of composer names included in the dataset
    '''

    folders = glob('%s/*' %datapath)
    X_train = []
    Y_train = []

    for folder in folders:
        files = glob('%s\\*.jpg' %folder)
        print('working on composer: %s' %(folder.split('\\')[-1]))
        for f in files:
            im = ndimage.imread(f, mode='L')
            im = im/255
            im = im.reshape(im.shape[0], im.shape[1], 1)
            X_train.append(im)
            Y_train.append(composers.index(folder.split('\\')[-1]))

    return np.asarray(X_train), np.asarray(Y_train)

if __name__ == '__main__':
    print('setting model')
    model = ResNet50.ResNet50(input_shape = (70, 400, 1), classes = 7)

    epochs = 100
    learning_rate = 0.001
    lr_decay = 0.001/100

    print('compiling model...')
    #optimizer_instance = Adam(lr=learning_rate, decay=lr_decay)#lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=0.001)
    #optimizer_instance = Adamax(lr=learning_rate, decay=lr_decay)
    optimizer_instance = SGD(lr=learning_rate, decay=lr_decay)
    #optimizer_instance = RMSprop(lr=learning_rate, decay=lr_decay)

    model.compile(optimizer=optimizer_instance, loss='categorical_crossentropy', metrics=['acc'])

    print('loading dataset......')
    composers = ['Bach', 'Beethoven', 'Brahms', 'Chopin', 'Grieg', 'Liszt', 'Mozart']
    datapath = 'Dataset_Train_Medium/'
    X_train, Y_train = load_dataset(datapath, composers)

    datapath_val = 'Dataset_Dev_Medium/'
    X_test, Y_test = load_dataset(datapath_val, composers)

    print('applying one-hot-encoding')
    Y_train = convert_to_one_hot(Y_train, 7).T
    Y_test = convert_to_one_hot(Y_test, 7).T

    print('setting up callbacks...')
    nancheck = callbacks.TerminateOnNaN()
    filepath = 'Models/weights-improvement-{epoch:02d}-{acc:.2f}.hdf5'
    saver = callbacks.ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=False, mode='max', period=1)
    logger = callbacks.CSVLogger('model-weights/trainingresults.log')
    callbacklist = [nancheck, saver, logger]

    print('starting model fitting')
    model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs=epochs, batch_size=72, callbacks=callbacklist)

    print('Saving model.........')
    model.save('second_run.h5')