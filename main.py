import numpy as np
import tensorflow as tf
import os
import tensorflow.keras.regularizers as regularizers
import csv
import pathlib
import cv2

# the model is the CNN while the secondary or second is the Inception model which was not used in the end.

checkpoint_path = "training_0/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print('getting training set')
split = 0.33
directory = "C:\py_proj\ML_Vision_Project\CV\train_set"


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\py_proj\ML_Vision_Project\CV\train_set",
    labels='inferred',
    color_mode='rgb',
    label_mode='int',
    validation_split=split,
    subset='training',
    seed=123,
    batch_size=16,
    image_size=(100, 100)
)
print('getting validation set')
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\py_proj\ML_Vision_Project\CV\train_set",
    labels='inferred',
    color_mode='rgb',
    label_mode='int',
    validation_split=split,
    subset='validation',
    seed=123,
    batch_size=16,
    image_size=(100, 100)
)

print(train_ds.class_names)
classnames = train_ds.class_names

csv_path = ":D\py_proj\ML_Vision_Project\CV\submission.csv"
location_gen = r"D:\py_proj\ML_Vision_Project\CV\test_set\{}"


def flipper(image, label):
    image = tf.image.flip_left_right(image)
    return image, label


augment_train = train_ds.map(
    flipper
)

train_ds = augment_train.concatenate(
    train_ds
)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
shapes = 0

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

numClasses = 5

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(100, 100, 3)),
    normalization_layer,
    tf.keras.layers.ZeroPadding2D(),
    tf.keras.layers.Conv2D(32, 5, 2, padding='valid', activation='relu',
                           kernel_regularizer=regularizers.L2(0.001)),
    tf.keras.layers.MaxPooling2D(3, strides=None, padding='same'),
    tf.keras.layers.Conv2D(64, 3, 1, padding='valid', activation='relu',
                           kernel_regularizer=regularizers.L2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, 3, 2, padding='valid', activation='relu',
                           kernel_regularizer=regularizers.L2(0.001)),
    tf.keras.layers.MaxPooling2D(3, strides=None, padding='valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(rate=0.2, noise_shape=(1024,)),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.L2(0.001)),
    tf.keras.layers.Dropout(rate=0.2, noise_shape=(512,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.L2(0.001)),
    tf.keras.layers.Dropout(rate=0.2, noise_shape=(100,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.1, noise_shape=(50,), ),
    tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=regularizers.L2(0.001)),
    tf.keras.layers.Dropout(rate=0.1, noise_shape=(25,)),
    tf.keras.layers.Dense(numClasses, activation='softmax'),
],
)

model.load_weights(checkpoint_path)

# the inception blocks each block is different
def inception_one(X_input):
    X = X_input
    sequence_six_one = tf.keras.Sequential([
        tf.keras.layers.Conv2D(3, 1, padding='valid', kernel_regularizer=regularizers.L2()),
        tf.keras.layers.Conv2D(32, 5, padding='valid', kernel_regularizer=regularizers.L2()),
        tf.keras.layers.MaxPooling2D()
    ])
    sequence_six_two = tf.keras.Sequential([
        tf.keras.layers.Conv2D(4, 1, padding='valid', kernel_regularizer=regularizers.L2()),
        tf.keras.layers.Conv2D(32, 5, padding='valid', kernel_regularizer=regularizers.L2()),
        tf.keras.layers.MaxPooling2D()
    ])
    sequence_six_three = tf.keras.Sequential([
        tf.keras.layers.Conv2D(2, 1, kernel_regularizer=regularizers.L2()),
        tf.keras.layers.Conv2D(64, 5, kernel_regularizer=regularizers.L2()),
        tf.keras.layers.MaxPooling2D()
    ])
    X = tf.concat([sequence_six_one(X), sequence_six_two(X), sequence_six_three(X)], -1)
    return X
def inception_two(X_input):
    X = X_input
    sequence_one = tf.keras.Sequential([
        tf.keras.layers.Conv2D(3, 1, kernel_regularizer=regularizers.L2()),
        tf.keras.layers.Conv2D(32, 3, kernel_regularizer=regularizers.L2()),
        tf.keras.layers.Conv2D(32, 5, kernel_regularizer=regularizers.L2())
    ])
    sequence_two = tf.keras.Sequential([
        tf.keras.layers.Conv2D(2, 1, kernel_regularizer=regularizers.L2()),
        tf.keras.layers.Conv2D(64, 5, kernel_regularizer=regularizers.L2()),
        tf.keras.layers.Conv2D(64, 3, kernel_regularizer=regularizers.L2())
    ])
    sequence_three = tf.keras.Sequential([
        tf.keras.layers.Conv2D(1, 1, kernel_regularizer=regularizers.L2()),
        tf.keras.layers.Conv2D(64, 3, kernel_regularizer=regularizers.L2()),
        tf.keras.layers.ZeroPadding2D(),
        tf.keras.layers.Conv2D(32, 7, kernel_regularizer=regularizers.L2())
    ])
    return tf.concat([sequence_two(X), sequence_three(X), sequence_one(X)], -1)
def inception_three(X_input):
    X = X_input
    sequence_six_one = tf.keras.Sequential([
        tf.keras.layers.Conv2D(3, 1, padding='valid', kernel_regularizer=regularizers.L2()),
        tf.keras.layers.Conv2D(32, 5, padding='valid', kernel_regularizer=regularizers.L2()),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), )
    ])
    sequence_six_two = tf.keras.Sequential([
        tf.keras.layers.Conv2D(4, 1, padding='valid', kernel_regularizer=regularizers.L2()),
        tf.keras.layers.Conv2D(64, 3, padding='valid', kernel_regularizer=regularizers.L2()),
        tf.keras.layers.MaxPooling2D(pool_size=(7, 7), strides=(1, 1))
    ])
    sequence_six_three = tf.keras.Sequential([
        tf.keras.layers.Conv2D(1, 1, padding='valid', kernel_regularizer=regularizers.L2()),
        tf.keras.layers.Conv2D(64, 7, kernel_regularizer=regularizers.L2()),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1))
    ])
    X = tf.concat([sequence_six_one(X), sequence_six_two(X), sequence_six_three(X)], -1)
    return X


def second():
    X_input = tf.keras.Input((100, 100, 3))
    X = X_input
    X = inception_one(X)
    X = inception_one(X)
    X = inception_three(X)
    X = inception_two(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Flatten()(X)
    print(X.shape)
    X_res = tf.keras.layers.Dense(50, activation='relu')(X)
    X = tf.keras.layers.Dropout(rate=0.4, noise_shape=(X.shape[1],))(X)
    X = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.L2(0.001))(X)
    X = tf.keras.layers.Dropout(rate=0.3, noise_shape=(512,))(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(50, activation=None)(X)
    X = tf.keras.layers.Dropout(0.4, noise_shape=(50,))(X)
    X = tf.keras.layers.ReLU()(tf.add(X, X_res))
    X = tf.keras.layers.Dense(25, activation='relu', kernel_regularizer=regularizers.L2(0.001))(X)
    X = tf.keras.layers.Dropout(rate=0.3, noise_shape=(25,))(X)
    X = tf.keras.layers.Dense(numClasses, activation='softmax')(X)
    return tf.keras.Model(inputs=X_input, outputs=X)


secondary = second()
secondary.load_weights(checkpoint_path)


# model.compile(optimizer='adam',
#               loss=tf.losses.sparse_categorical_crossentropy,
#               metrics=['accuracy'])

converter = tf.lite.TFLiteConverter.from_keras_model(model)
lite_model = converter.convert()

with tf.io.gfile.GFile('first_save.tflite', 'wb') as f:
    f.write(lite_model)


# secondary.compile(optimizer='adam',
#                   loss=tf.losses.sparse_categorical_crossentropy,
#                   metrics=['accuracy'])


# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1,
#                                                  )
# secondary.fit(train_ds, batch_size=2, validation_data=test_ds, epochs=30, callbacks=[cp_callback])
# model.fit(train_ds, batch_size=2, validation_data=test_ds, epochs=48, callbacks=[cp_callback])

# this is where the submission.csv file is updated and predictions are made
strict = 0.75
later = []
# with open(csv_path, 'rt') as f:
#     data = csv.reader(f)
#
#     ct = 0
#     for row in data:
#         later.append(row)
#         if ct == 0:
#             print('nothing')
#             ct = 1
#         else:
#             name = row[0]
#             print(name)
#             fin = location_gen.format(name)
#             path = pathlib.Path(fin)
#             print(path)
#             again = cv2.imread(fin)
#             again = tf.image.resize(again, (100, 100))
#             again = np.expand_dims(again, axis=0)
#             prediction = model.predict(again)
#             stor = 10
#             for i in range(len(classnames)):
#                 if prediction[0][i] > strict:
#                     stor = i
#                     break
#
#             if stor != 10:
#                 print('\n')
#                 print(classnames[stor])
#                 row[1] = classnames[stor]
#                 print('\n')
#             else:
#                 print('\n')
#                 print('unknown')
#                 print('\n')
#             print(classnames)
#             print(prediction)
#             print('---------------------------------')
# with open("C:\py_proj\ML_Vision_Project\CV\submission.csv", 'w', newline='') as csv_file:
#     csv_writer = csv.writer(csv_file)
#     for row in later:
#         csv_writer.writerow([row[0], row[1]])
#     print('no prob')