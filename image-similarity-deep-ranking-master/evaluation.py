
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Embedding
import cv2

def convnet_model_():
    vgg_model = VGG16(weights=None, include_top=False)
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Lambda(lambda x_: K.l2_normalize(x, axis=1))(x)
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model


def deep_rank_model():
    convnet_model = convnet_model_()
    first_input = Input(shape=(224, 224, 3))
    first_conv = Conv2D(96, kernel_size=(8, 8), strides=(16, 16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3, 3), strides=(4, 4), padding='same')(first_conv)
    first_max = Flatten()(first_max)
    first_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_max)

    second_input = Input(shape=(224, 224, 3))
    second_conv = Conv2D(96, kernel_size=(8, 8), strides=(32, 32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7, 7), strides=(2, 2), padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_max)

    merge_one = concatenate([first_max, second_max])

    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(4096)(merge_two)
    l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)

    final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    return final_model

def get_model(weights_path):
    model = deep_rank_model()
    model.load_weights(weights_path)
    return model

def write_errors_similarities(erros_dir_path, query, pos, neg, count, imgs_path):
    erros_query = erros_dir_path+str(count)+"_query_"+query
    erros_pos = erros_dir_path +str(count)+"_pos_"+ pos
    erros_neg = erros_dir_path +str(count)+"_neg_"+ neg

    image1 = cv2.imread(imgs_path + query)
    image2 = cv2.imread(imgs_path + pos)
    image3 = cv2.imread(imgs_path + neg)

    cv2.imwrite(erros_query, image1)
    cv2.imwrite(erros_pos, image2)
    cv2.imwrite(erros_neg, image3)

def evaluate(triplets_path,imgs_path,weights_path):
    model=get_model(weights_path)
    count=0
    tp=0
    acc=0
    all=0
    f = open(triplets_path)
    f_read = f.read()
    for line in f_read.split('\n'):
        if len(line) > 1:
            all=all+1
            query=line.split(',')[0]
            positive=line.split(',')[1]
            negative=line.split(',')[2]

            image1 = load_img(imgs_path+query)
            image1 = img_to_array(image1).astype("float64")
            image1 = transform.resize(image1, (224, 224))
            image1 *= 1. / 255
            image1 = np.expand_dims(image1, axis=0)

            embedding1 = model.predict([image1, image1, image1])[0]

            image2 = load_img(imgs_path+positive)
            image2 = img_to_array(image2).astype("float64")
            image2 = transform.resize(image2, (224, 224))
            image2 *= 1. / 255
            image2 = np.expand_dims(image2, axis=0)

            embedding2 = model.predict([image2, image2, image2])[0]

            image3 = load_img(imgs_path+negative)
            image3 = img_to_array(image3).astype("float64")
            image3 = transform.resize(image3, (224, 224))
            image3 *= 1. / 255
            image3 = np.expand_dims(image3, axis=0)

            embedding3 = model.predict([image3, image3, image3])[0]

            distance1 = sum([(embedding1[idx] - embedding2[idx]) ** 2 for idx in range(len(embedding1))]) ** (0.5)
            distance2 = sum([(embedding1[idx] - embedding3[idx]) ** 2 for idx in range(len(embedding1))]) ** (0.5)
            # squared_euclidian_distance = sum([(embedding1[idx] - embedding2[idx])**2 for idx in range(len(embedding1))])
            print("distance poz ", distance1, "distance neg ", distance2, "\n")
            if distance1 < distance2:
                tp = tp + 1
            else:
                count=count+1
                write_errors_similarities(ERRORS_PATH, query,positive,negative,count,imgs_path)
                print(query, positive, negative)

    acc = (tp * 100) / all
    print("Acc on validation set: ", acc)
    f.close()


TRIPLETS_PATH_VALID= "G:\\LicentaSimona\\repo\\similarity\\image-similarity-deep-ranking-master\\tripletsvalid.csv"
IMAGES_PATH="G:\\LicentaSimona\\dataset\\similarity\\Dress\\"
WEIGHTS_PATH="G:\LicentaSimona\\repo\similarity\\image-similarity-deep-ranking-master\\weights\\weights12-23AM-January-10-2019"
ERRORS_PATH="G:\\LicentaSimona\\dataset\\errors_similarity\\"
evaluate(TRIPLETS_PATH_VALID, IMAGES_PATH, WEIGHTS_PATH)



