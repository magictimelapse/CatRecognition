
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input,  Dropout
from plotly.graph_objs import Data, Scatter
from keras.models import load_model
import plotly


from keras import backend as K
img_width, img_height = 299, 299
train_data_dir = "set your path to trainings data here"
validation_data_dir = "set your path to test data here"

batch_size = 6
initial_epoch = 0
endepochs = 10

model_name = 'inception_v3_1'

if initial_epoch > 0:
    refit = True
else:
    refit = False

def add_last_layer(base_model, nb_classes = 2):
    x = base_model.output
    #x = Dropout(0.4)(x)
    x = Dense(16,activation="relu")(x)
    #x = Dropout(0.4)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model
def exchange_last_layers(base_model, nb_classes=2):
    x = base_model.output
    x= GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(4,activation="relu")(x)
    x = Dropout(0.4)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


if not refit:
    if model_name == 'inception_v3_1' or model_name == 'inception_v3_2'or model_name == "InceptionResNet_V2_0":
        if (K.image_dim_ordering() == 'th'):
            input_tensor = Input(shape=(3, 299, 299))
        else:
            input_tensor = Input(shape=(299, 299, 3))
        if model_name == 'inception_v3_1':
            base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet',
                                 include_top=False)
            mymodel = exchange_last_layers(base_model)
        if model_name == 'inception_v3_2':
            base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet',
                                 include_top=True)
            mymodel = add_last_layer(base_model)

        elif model_name == "InceptionResNet_V2_0":
            base_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet',
                                     include_top=False)
            mymodel = exchange_last_layers(base_model)


        for layer in base_model.layers:
            layer.trainable = False
        mymodel.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


    elif model_name == 'simple_inception':
        mymodel = model.build_simple_inception_model((img_width, img_height))
    elif model_name == "inception_identity3":
        mymodel = model.build_inception_identity_model3((img_width, img_height))

if refit:
    mymodel = load_model(model_name+'.h5')
# Initiate the train and test generators with data Augumentation
print(mymodel.summary())
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

# Save the model according to the conditions
checkpoint = ModelCheckpoint(model_name+'.h5', monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

class_dictionary = train_generator.class_indices
# set class weights. Cats is outnumbered by non-cat by about 10:1
class_weights = {class_dictionary['cats']:0.85,class_dictionary['noCat']:0.15}
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
    plotly.offline.plot(data, filename='history.html')
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
        plotly.offline.plot(data, filename='history_accuracy.html')