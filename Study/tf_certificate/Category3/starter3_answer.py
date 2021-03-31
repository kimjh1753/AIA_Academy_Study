# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail. 
#
# NOTE THAT THIS IS UNLABELLED DATA. 
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.

import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()

    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(
    # YOUR CODE HERE)
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=1.2, 
        zoom_range=0.7,
        fill_mode='nearest'
    )
    train_generator = training_datagen.flow_from_directory( 
        # YOUR CODE HERE
        TRAINING_DIR,
        target_size=(150, 150),
        class_mode='categorical'
    )

    model = tf.keras.models.Sequential([
    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(100, 64, padding='same', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    rl = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)
    history = model.fit_generator(train_generator, epochs=10, callbacks=[es, rl], 
                                  steps_per_epoch=32, validation_steps=4,
                                  validation_data=train_generator,
                                  verbose=1)

    # loss = history.history['loss']
    # acc = history.history['acc']
    
    # print("loss : ", loss[-1])                        
    # print("acc : ", acc[-1])

    # loss :  1.1145190000534058
    # acc :  0.3720703125

    loss, acc = model.evaluate(train_generator)
    print('loss : ', loss)
    print('acc : ', acc)

    # loss :  45.34305953979492
    # acc :  0.33095237612724304

    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
