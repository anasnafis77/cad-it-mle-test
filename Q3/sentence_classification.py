import os
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from matplotlib import pyplot as plt

def build_dataset(data_dir):
    file_list = os.listdir(data_dir)
    labels = []
    sentences = []
    i = 0
    for filename in file_list:
        f = open(data_dir + filename, 'r')
        for line in f:
            if '###' not in line:
                label_pat = 'MISC|AIMX|OWNX|CONT|BASE'
                sentence_pat = '[^\t-]*'
                label = re.findall(label_pat, line)[0]
                sentence = ''.join(re.findall(sentence_pat, line.strip(label)))
                labels.append(label)
                sentences.append(sentence)
                # saving data
                path = data_dir + '/dataset/' +label
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                file = open(path + f'/{label}_{i}.txt', 'w')
                file.write(sentence)
                i += 1
    
    data_df = pd.DataFrame({'sentence': sentences, 'label': labels})
    return data_df

# def text_cleaning(text):
#     # case folding
#     # 1. lowering
#     text = text.lower()
#     # 2. Remove number
#     text = re.sub(r'\d+', '', text)
#     # Remove punctuation
#     text = text.translate(str.maketrans('','',string.punctuation))
#     return text

# def tokenize(text):
#     text = text_cleaning(text)
#     stemmer = PorterStemmer()
#     # tokenizing
#     tokenized = word_tokenize(text)
#     # stemming word in english
#     stop_words = stopwords.words('english')
#     stemmed = [stemmer.stem(word) for word in tokenized if word not in stop_words]
#     return stemmed

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

def load_dataset(ds_path):
    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        ds_path,
        label_mode='categorical', 
        batch_size=batch_size, 
        validation_split=0.2, 
        subset='training', 
        seed=seed)
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        ds_path, 
        batch_size=batch_size,
        label_mode='categorical', 
        validation_split=0.2, 
        subset='validation', 
        seed=seed)
    
    max_features = 7500
    sequence_length = 100
    vectorizer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)
    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorizer(text), label
    
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorizer.adapt(train_text)

    train_ds = raw_train_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)
    
    return train_ds, test_ds, vectorizer
    
def build_model(num_class):
    embedding_dim = 100
    max_features = 7500
    model = tf.keras.Sequential([
                layers.Embedding(max_features + 1, embedding_dim),
                layers.Dropout(0.2),
                layers.GlobalAveragePooling1D(),
                layers.Dropout(0.2),
                layers.Dense(num_class, activation='softmax')])

    model.compile(loss=losses.CategoricalCrossentropy(),
              optimizer='adam',
              metrics='acc')
    print(model.summary())
    return model


def model_training(num_class):
    ds_dir = 'Q3/data/dataset/'
    train_ds, test_ds, vectorizer = load_dataset(ds_dir)

    model = build_model(num_class)
    epochs = 20
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs)
    history = history.history
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(epochs, loss, 'bo', label='Training loss')
    ax[0].plot(epochs, val_loss, 'b', label='Validation loss')
    ax[0].set_title('Training and validation loss')
    ax[1].plot(epochs, acc, 'bo', label='Training acc')
    ax[1].plot(epochs, val_acc, 'b', label='Validation acc')
    ax[1].set_title('Training and validation accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(loc='lower right')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend(loc='lower right')

    plt.show()
    return model, vectorizer

    
if __name__ == '__main__':
    data_dir = 'Q3/data/'
    num_class = 5
    if not os.path.exists(data_dir + 'dataset'):
        build_dataset(data_dir)
    model, vectorizer = model_training(num_class)
    # saving model
    model_path = 'Q3/model.h5'
    model.save(model_path)






    
