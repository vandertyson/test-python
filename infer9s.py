import numpy as np
from keras.models import Model
from keras.engine.topology import merge as merge_l
from keras.layers import (
    Input, Convolution2D, MaxPooling2D, UpSampling2D,
    Reshape, core, Dropout,
    Activation, BatchNormalization)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from keras import backend as K
import tables as tb


FN_WEIGHTS = "/data/models/working/v9s/AOI_3_Paris_val_weights_last.h5"
FN_IM  = "/data/test/AOI_3_Paris/RGB-PanSharpen/RGB-PanSharpen_AOI_3_Paris_img522.tif"

def get_unet():
    conv_params = dict(activation='relu', border_mode='same')
    merge_params = dict(mode='concat', concat_axis=1)
    inputs = Input((8, 256, 256))
    conv1 = Convolution2D(32, 3, 3, **conv_params)(inputs)
    conv1 = Convolution2D(32, 3, 3, **conv_params)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, **conv_params)(pool1)
    conv2 = Convolution2D(64, 3, 3, **conv_params)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, **conv_params)(pool2)
    conv3 = Convolution2D(128, 3, 3, **conv_params)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, **conv_params)(pool3)
    conv4 = Convolution2D(256, 3, 3, **conv_params)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, **conv_params)(pool4)
    conv5 = Convolution2D(512, 3, 3, **conv_params)(conv5)

    up6 = merge_l([UpSampling2D(size=(2, 2))(conv5), conv4], **merge_params)
    conv6 = Convolution2D(256, 3, 3, **conv_params)(up6)
    conv6 = Convolution2D(256, 3, 3, **conv_params)(conv6)

    up7 = merge_l([UpSampling2D(size=(2, 2))(conv6), conv3], **merge_params)
    conv7 = Convolution2D(128, 3, 3, **conv_params)(up7)
    conv7 = Convolution2D(128, 3, 3, **conv_params)(conv7)

    up8 = merge_l([UpSampling2D(size=(2, 2))(conv7), conv2], **merge_params)
    conv8 = Convolution2D(64, 3, 3, **conv_params)(up8)
    conv8 = Convolution2D(64, 3, 3, **conv_params)(conv8)

    up9 = merge_l([UpSampling2D(size=(2, 2))(conv8), conv1], **merge_params)
    conv9 = Convolution2D(32, 3, 3, **conv_params)(up9)
    conv9 = Convolution2D(32, 3, 3, **conv_params)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
    adam = Adam()

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy', jaccard_coef, jaccard_coef_int])
    return model


def jaccard_coef(y_true, y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def infer():    
    # model = get_unet()
    # model.load_weights(FN_WEIGHTS)
    with tb.open_file(FN_IM, 'r') as f:
            print(f)
    #         slice_pos = 5
    #         slice_id = image_id + '_' + str(slice_pos)
    #         im = np.array(f.get_node('/' + slice_id))
    #         im = np.swapaxes(im, 0, 2)
    #         im = np.swapaxes(im, 1, 2)
    #         X_train.append(im)
    # X_test = 
    # y_pred = model.predict(X_test, batch_size=1, verbose=1)

infer()    