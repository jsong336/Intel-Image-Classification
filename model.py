# Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from CONST import IMG_SHAPE, BATCH_SIZE, NUM_DATA, NUM_TEST, EPOCHS, PATH_TRAIN, PATH_TEST

import os


def get_datasets_gen():
        # data preprocess
        train_idg = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
        test_idg = ImageDataGenerator( rescale = 1./255,)

        train_gen = train_idg.flow_from_directory(
                PATH_TRAIN,
                target_size = IMG_SHAPE,
                batch_size = BATCH_SIZE,
                class_mode = 'categorical')

        test_gen = test_idg.flow_from_directory(
                PATH_TEST,
                target_size = IMG_SHAPE,
                batch_size = BATCH_SIZE,
                class_mode='categorical')

        return train_gen, test_gen


def build_model():
        input_shape = (IMG_SHAPE[0], IMG_SHAPE[1], 3)

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size = (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Conv2D(filters=64, kernel_size = (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))

        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=6, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model


def fit_model(model, train_gen, test_gen):
        return model.fit_generator(
                generator=train_gen, 
                steps_per_epoch=NUM_DATA, 
                validation_data=test_gen,
                validation_steps=NUM_TEST,
                epochs=EPOCHS)


if __name__=='__main__':
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        
        train_set, test_set = get_datasets_gen()
        model = build_model()
        fit_model(model, train_set, test_set)
        model.save('model.h5')
        
# save https://www.tensorflow.org/guide/keras/save_and_serialize