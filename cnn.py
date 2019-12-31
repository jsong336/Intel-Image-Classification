# Model
# save https://www.tensorflow.org/guide/keras/save_and_serialize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

IMG_SHAPE = (128, 128)
BATCH_SIZE = 32

NUM_DATA = 2000 
NUM_TEST = 200
EPOCHS = 1
PATH = 'intel-image-classification/'
PATH_TRAIN = PATH + 'seg_train/seg_train'
PATH_TEST = PATH + 'seg_test/seg_test'


"""
Ubuntu NVIDA => sad life
NUM_DATA = 14034 
NUM_TEST = 3000
EPOCHS = 50
"""

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


model = Sequential()

model.add(Conv2D(filters=32, kernel_size = (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(filters=64, kernel_size = (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=6, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
        generator=train_gen, 
        steps_per_epoch=NUM_DATA, 
        validation_data=test_gen,
        validation_steps=NUM_TEST,
        epochs=EPOCHS)


loss, evaluate = model.evaluate_generator(test_gen, steps=NUM_TEST)
# loss = 0.5321 & evaluate = 0.8156
# model.save('n2000e1b32.h5')