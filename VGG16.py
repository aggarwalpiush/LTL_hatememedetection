from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import VGG16
from keras.models import Model

import numpy as np

import os

from sister.tokenizers import SimpleTokenizer
from sister.word_embedders import FasttextEmbedding
import easyocr
import sys

word_embedder = FasttextEmbedding("en")
tokenizer = SimpleTokenizer()
vgg_model = VGG16()
vgg_model = Model(inputs = vgg_model.inputs, outputs = vgg_model.layers[-2].output)

def extract_text_features(word_embedder, tokenizer, sentence):
    tokens = tokenizer.tokenize(sentence)
    text_features = word_embedder.get_word_vectors(tokens)
    text_features = np.mean(text_features, axis=0)
    return text_features

def extract_image_features(img_path, img_model):
    img = load_img(img_path, target_size=(224,224))
    img = np.array(img)
    reshaped_img = img.reshape(1,224,224,3)
    imgx = preprocess_input(reshaped_img)
    img_features = img_model.predict(imgx, use_multiprocessing=True)
    img_features = img_features.reshape((4096,))
    return img_features

def extract_features(word_embedder, tokenizer, sentence, img_path, img_model):
    text_features = extract_text_features(word_embedder, tokenizer, sentence)
    image_features = extract_image_features(img_path, img_model)
    return np.concatenate((text_features, image_features), axis=0)

def train_model():
    clf = Sequential()
    clf.add(Dense(1000, kernel_initializer="uniform", input_shape=(4396,)))
    clf.add(Dropout(0.5))
    clf.add(Dense(500))
    clf.add(Dropout(0.5))
    clf.add(Dense(100))
    clf.add(Dropout(0.5))
    clf.add(Dense(50))
    clf.add(Dropout(0.5))
    clf.add(Dense(2, activation="softmax"))

    clf.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    X_id, X_train, Y_train = pickle.load(open("train_dev.pkl", "rb"))

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    Y_train = to_categorical(Y_train)

    from sklearn.model_selection import train_test_split

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.7, random_state=42)
    file_path = f"../hate_meme_detection_model.hdf5"
    if not os.path.exists(file_path):
        clf.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=32)
    clf.load_weights(file_path)
    return clf


def predict_meme_class(clf, meme_path):

    reader = easyocr.Reader(['en']) # need to run only once to load model into memory
    sentence = reader.readtext(meme_path)


    x_text = extract_features(word_embedder, tokenizer, sentence, meme_path, vgg_model)

    predict_label = clf.predict(x_text)

    return predict_label


def main():
    image_path = sys.argv[1]
    clf = train_model()
    print(predict_meme_class(image_path, clf)

if __name__ == '__main__':
    main()



