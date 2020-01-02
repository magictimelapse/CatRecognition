#!/usr/bin/env python
import models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import preprocess_input as inceptionpreprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16preprocess_input
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.nasnet import preprocess_input as nasnetpreprocess_input
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD

from keras.layers import Dense, GlobalAveragePooling2D, Input,  Dropout, AveragePooling2D, Conv2D, Concatenate, Flatten, MaxPooling2D
from plotly.graph_objs import Data, Scatter
from keras.models import load_model
import plotly
import tensorflow as tf
import numpy as np
from keras.utils.vis_utils import plot_model
### some GTX 2070 nonsense ###
use_GTX_2070 = False
if use_GTX_2070:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
###

from keras import backend as K


def preprocess_input_vgg(x):
    X = np.expand_dims(x, axis=0)
    #X /= 255.
    X = vgg16preprocess_input(X)
    return X[0]

def weighted_accuracy(y_true,y_pred):
    weights = tf.convert_to_tensor(np.array([0.95,0.05]))

    cnt=K.sum(weights)
    err=K.sum(K.not_equal(K.argmax(y_pred,axis=-1)*weights,K.argmax(y_true,axis=-1)*weights))
    acc=1.0-(err/cnt)
    return acc



def add_last_layer(base_model, nb_classes = 2):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(16,activation="relu")(x)

    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


def exchange_last_layers(base_model, nb_classes=2):
    x = base_model.output
    x= GlobalAveragePooling2D()(x)
    x = Dropout(0.6)(x)

    x = Dense(4,activation="relu")(x)
    x = Dropout(0.6)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

def exchange_last_layers_nasnet(base_model,nb_classes=2):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.6)(x)
    x = Dense(2,activation="relu")(x)
    #x = Dense(4, activation="relu")(x)
    x = Dropout(0.4)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


def exchange_last_layers_nasnet2(base_model,nb_classes=2):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.6)(x)
    x = Dense(64,activation="relu")(x)
    #x = Dense(4, activation="relu")(x)
    x = Dropout(0.6)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

def inception_layer(input_img):
    tower_1 = Conv2D(1, (1, 1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(1, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(1, (1, 1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(1, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    tower_3 = Conv2D(1, (1, 1), padding='same', activation='relu')(tower_3)

    output = Concatenate()([tower_1, tower_2, tower_3])
    #output = Flatten()(output)
    return output

def exchange_last_layers2(base_model, nb_classes=2):
    base_model.layers.pop()
    x = base_model.output
    x = inception_layer(x)
    x= GlobalAveragePooling2D()(x)
    x = Dropout(0.6)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

def add_last_layer_inception31c(base_model, nb_classes=2):
    x = base_model.output
    x= Dropout(0.5)(x)
    predictions = Dense(nb_classes,activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

def main():
    img_width, img_height = 224, 224  # vgg16
    train_data_dir = r"" ### Set here the train data directory
    validation_data_dir = r"" ### Set here the validation data directory

    batch_size = 32
    initial_epoch = 0
    endepochs = 133

    # model_name = 'inception_v3_1b'
    # model_name = "nasnet"
    # model_name = 'simple'

    # model_name = 'nasnet2'
    #model_name = 'inception_v3_1c'
    model_name = 'vgg16_1'
    if initial_epoch > 0:
        refit = True
    else:
        refit = False


    if not refit:
        if True: #model_name == 'inception_v3_1' or model_name == 'inception_v3_2'or model_name == "InceptionResNet_V2_0" or model_name =="inception_v3_1b" or model_name == "nasnet" :
            if (K.image_dim_ordering() == 'th'):
                input_tensor = Input(shape=(3, img_width, img_height))
            else:
                input_tensor = Input(shape=(img_width, img_height, 3))
            if model_name == 'inception_v3_1':
                base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet',
                                     include_top=False)
                mymodel = exchange_last_layers(base_model)
            if model_name == 'inception_v3_1b':
                base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet',
                                     include_top=False)
                mymodel = exchange_last_layers2(base_model)
            if model_name == 'inception_v3_2':
                bbase_model = InceptionV3(input_tensor=input_tensor, weights='imagenet',
                                     include_top=False)
                mymodel = add_last_layer(base_model)

            if model_name == 'nasnet':
                base_model = NASNetMobile(input_tensor=input_tensor, weights='imagenet',include_top=False)
                mymodel = exchange_last_layers_nasnet(base_model)

            if model_name == 'nasnet2':
                    base_model = NASNetMobile(input_tensor=input_tensor, weights='imagenet', include_top=False)
                    mymodel = exchange_last_layers_nasnet2(base_model)
            if model_name == "InceptionResNet_V2_0":
                base_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet',
                                         include_top=False)
                mymodel = exchange_last_layers(base_model)

            if model_name == 'inception_v3_1c':
                temp_model = InceptionV3(input_tensor=input_tensor, weights='imagenet',
                                         include_top=True)
                temp_model.layers.pop()
                base_model =  Sequential()
                for layer in temp_model:
                    base_model.add(layer)

                my_model = add_last_layer_inception31c(base_model)

            if model_name == 'vgg16_1':

                temp_model = VGG16(input_tensor=input_tensor, weights='imagenet',
                                         include_top=False,pooling = 'avg')

                base_model =  Sequential()
                for layer in temp_model.layers:
                    base_model.add(layer)

                mymodel = add_last_layer_inception31c(base_model) #use the same function as above

            #for layer in base_model.layers[:-5]: #block 5
            for layer in base_model.layers[:-9]:  # block 5
                layer.trainable = False


            if model_name == 'simple':
                mymodel = models.build_simple_model(input_tensor)


        optimizer = SGD(lr=1e-5, nesterov=True,momentum = 0.9,decay=1e-5/20.)
        mymodel.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])


    if refit:
        mymodel = load_model(model_name+'.h5')
    # Initiate the train and test generators with data Augumentation
    print(mymodel.summary())


    print('Number of trainable weights: ', len(mymodel.trainable_weights))
    train_datagen = ImageDataGenerator(
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
        rotation_range=20,

        preprocessing_function=preprocess_input_vgg
        )

    test_datagen = ImageDataGenerator(
    horizontal_flip = True,
    fill_mode = "nearest",
    zoom_range = 0.3,
    width_shift_range = 0.3,
    height_shift_range=0.3,
       rotation_range=20,

    preprocessing_function=preprocess_input_vgg
    )

    train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
        shuffle = True,
    class_mode = "categorical")

    validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
        shuffle=True,
    target_size = (img_height, img_width),
    class_mode = "categorical")

    # Save the model according to the conditions
    checkpoint = ModelCheckpoint(model_name+'.h5', monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

    class_dictionary = train_generator.class_indices
    # set class weights. Cats is outnumbered by non-cat by about 10:1
    class_weights = {class_dictionary['cat']:0.5,class_dictionary['noCat']:0.5}
    print(class_dictionary)
    history= mymodel.fit_generator(
        train_generator,
        epochs = endepochs,
        initial_epoch =  initial_epoch,
        validation_data = validation_generator,
        class_weight=class_weights,
        steps_per_epoch=len(train_generator) ,
       validation_steps = len(validation_generator),
        callbacks=[checkpoint])
    plot_history = True
    if plot_history:
        trace = Scatter(
            x=list(range(len(list(history.history['loss'])))),
            y=list(history.history['loss'])
        )
        traceVal = Scatter(
            x=list(range(len(list(history.history['val_loss'])))),
            y=list(history.history['val_loss'])
        )
        data = [trace, traceVal]
        plotly.offline.plot(data, filename='history_{0}_{1}.html'.format(initial_epoch,endepochs))
        if True:
            trace = Scatter(
                x=list(range(len(list(history.history['acc'])))),
                y=list(history.history['acc'])
            )
            traceVal = Scatter(
                x=list(range(len(list(history.history['val_acc'])))),
                y=list(history.history['val_acc'])
            )
            data = [trace, traceVal]
            plotly.offline.plot(data, filename='history_accuracy_{0}_{1}.html'.format(initial_epoch,endepochs))


if __name__ == "__main__":
    main()