# coding: utf-8

from  __future__ import absolute_import
from __future__ import print_function
from ImageDataGeneratorCustom import ImageDataGeneratorCustom
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import os
import datetime

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

TRIPLETS_PATH_TRAIN= "G:\\LicentaSimona\\repo\\similarity\\image-similarity-deep-ranking-master\\triplets.csv"
TRIPLETS_PATH_VALID= "G:\\LicentaSimona\\repo\\similarity\\image-similarity-deep-ranking-master\\tripletsvalid.csv"
IMAGES_PATH="G:\\LicentaSimona\\dataset\\similarity\\"

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


class CustomTensorBoard(TensorBoard):
    """ to log the loss after each batch
    """

    def __init__(self, log_every=1, **kwargs):
        super(CustomTensorBoard, self).__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()

        super(CustomTensorBoard, self).on_batch_end(batch, logs)


def convnet_model_VGG():
    vgg_model = VGG16(weights=None, include_top=False)
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x) #no global
    x = Dense(4096, activation='relu')(x) # conv de size-ul rams x 4096
    x = Dropout(0.6)(x) #no dropout
    x = Dense(4096, activation='relu')(x) #conv de 1x1 si global reduce nr param
    x = Dropout(0.6)(x) #no dropout
    x = Lambda(lambda  x_: K.l2_normalize(x,axis=1))(x)
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model

def deep_rank_model():
 
    convnet_model = convnet_model_VGG()
    first_input = Input(shape=(224,224,3))
    first_conv = Conv2D(96, kernel_size=(8, 8),strides=(16,16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3,3),strides = (4,4),padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)

    second_input = Input(shape=(224,224,3))
    second_conv = Conv2D(96, kernel_size=(8, 8),strides=(32,32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7),strides = (2,2),padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])

    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

    final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    return final_model


deep_rank_model = deep_rank_model()

for layer in deep_rank_model.layers:
    print (layer.name, layer.output_shape)

model_path = "./deep_ranking"

class DataGenerator(object):
    def __init__(self, params, target_size=(224, 224)):
        self.params = params
        self.target_size = target_size
        self.idg = ImageDataGeneratorCustom(**params)

    #original functions
    # def get_train_generator(self, batch_size):-
    #     return self.idg.flow_from_directory("./dataset/",
    #                                         batch_size=batch_size,
    #                                         target_size=self.target_size,shuffle=False,
    #                                         triplet_path  ='./triplet_5033.txt'
    #                                        )

    # def get_test_generator(self, batch_size):
    #     return self.idg.flow_from_directory("./dataset/",
    #                                         batch_size=batch_size,
    #                                         target_size=self.target_size, shuffle=False,
    #                                         triplet_path  ='./triplet_5033.txt'
    #                                     )
    def get_train_generator(self, batch_size):
        return self.idg.flow_from_directory(IMAGES_PATH,
                                            batch_size=batch_size,
                                            target_size=self.target_size,
                                            shuffle=False,
                                            triplet_path  =TRIPLETS_PATH_TRAIN,
                                            classes=['Dress']
                                            )

    def get_valid_generator(self, batch_size):
        return self.idg.flow_from_directory(IMAGES_PATH,
                                            batch_size=batch_size,
                                            target_size=self.target_size,
                                            shuffle=False,
                                            triplet_path=TRIPLETS_PATH_VALID,
                                            classes=['Dress']
                                            )


dg = DataGenerator({
    "rescale": 1. / 255,
    "horizontal_flip": True,
    "vertical_flip": True,
    "zoom_range": 0.2,
    "shear_range": 0.2,
    "rotation_range": 30,
"fill_mode": 'nearest' 
}, target_size=(224, 224))

batch_size = 1
batch_size *= 3
train_generator = dg.get_train_generator(batch_size)
valid_generator = dg.get_valid_generator(batch_size)

_EPSILON = K.epsilon()
def _loss_tensor(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    loss =  tf.convert_to_tensor(0,dtype=tf.float32)
    g = tf.constant(1.0, shape=[1], dtype=tf.float32)
    for i in range(0,batch_size,3):
        try:
            q_embedding = y_pred[i+0]
            p_embedding =  y_pred[i+1]
            n_embedding = y_pred[i+2]
            D_q_p =  K.sqrt(K.sum((q_embedding - p_embedding)**2))
            D_q_n = K.sqrt(K.sum((q_embedding - n_embedding)**2))
            loss = (loss + g + D_q_p - D_q_n )            
        except:
            continue
    loss = loss/(batch_size/3)
    zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
    return tf.maximum(loss,zero)

#deep_rank_model.load_weights('deepranking.h5')
deep_rank_model.compile(loss=_loss_tensor, optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True))

train_images_triplets=4041
train_steps_per_epoch = int(train_images_triplets/batch_size)
valid_images_triplets=903
valid_steps_per_epoch = int(valid_images_triplets/batch_size)

train_epocs = 200

now=datetime.datetime.now().strftime("%I-%M%p-%B-%d-%Y")
weights_path="./weights/weights"+now

modelCheckpoint=ModelCheckpoint(weights_path,
                              save_best_only=True,
                              save_weights_only=True,
                              monitor='val_loss',
                              mode='min',
                              period=1)


log_path="./logs"
both_logs_path="./bothlogs"

tensorboard=CustomTensorBoard(log_dir=log_path,  write_graph=True, write_images=False)
earlyStopping=EarlyStopping(monitor='val_loss', patience=50, verbose=1, min_delta=1e-4, mode='min')
reduceLrOnPlateau=ReduceLROnPlateau(monitor='val_loss', patience=10, mode='min', min_lr=1e-6, factor=0.2)


deep_rank_model.fit_generator(train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=train_epocs,
                        validation_data=valid_generator,
                        validation_steps=valid_steps_per_epoch,
                        callbacks=[tensorboard, earlyStopping, modelCheckpoint, reduceLrOnPlateau,
                                   TrainValTensorBoard(log_dir=both_logs_path, write_graph=False)]
                        )

# model_path = "deepranking.h5"
# deep_rank_model.save_weights(model_path)
#f = open('deepranking.json','w')
#f.write(deep_rank_model.to_json())
#f.close()

from skimage import transform
def evaluate():
    model=deep_rank_model
    all=len(valid_generator.filenames)
    files=valid_generator.filenames
    tp=0
    for i in range(all,3):
        image1 = load_img(files[i])
        image1 = img_to_array(image1).astype("float64")
        image1 = transform.resize(image1, (224, 224))
        image1 *= 1. / 255
        image1 = np.expand_dims(image1, axis=0)

        embedding1 = model.predict([image1, image1, image1])[0]

        image2 =  load_img(files[i+1])
        image2 = img_to_array(image2).astype("float64")
        image2 = transform.resize(image2, (224, 224))
        image2 *= 1. / 255
        image2 = np.expand_dims(image2, axis=0)

        embedding2 = model.predict([image2, image2, image2])[0]

        image3 = load_img(files[i+2])
        image3 = img_to_array(image3).astype("float64")
        image3 = transform.resize(image3, (224, 224))
        image3 *= 1. / 255
        image3 = np.expand_dims(image3, axis=0)

        embedding3 = model.predict([image3, image3, image3])[0]

        distance1 = sum([(embedding1[idx] - embedding2[idx]) ** 2 for idx in range(len(embedding1))]) ** (0.5)
        distance2 = sum([(embedding1[idx] - embedding3[idx]) ** 2 for idx in range(len(embedding1))]) ** (0.5)
        # squared_euclidian_distance = sum([(embedding1[idx] - embedding2[idx])**2 for idx in range(len(embedding1))])
        print("distance poz ",distance1,"distance neg ",distance2,"\n")
        if distance1<distance2:
            tp=tp+1

    acc=(tp/100)*all
    print("Acc on validation set: ",acc)

#evaluate()