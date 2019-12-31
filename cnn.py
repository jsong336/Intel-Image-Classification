from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SHAPE = (128, 128)
BATCH_SIZE = 32
NUM_DATA = 14034
NUM_TEST = 3000
# data preprocess
train_idg = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_idg = ImageDataGenerator( rescale = 1./255,)

train_gen = train_idg.flow_from_directory(
        'intel-image-classification/seg_train/seg_train',
        target_size = IMG_SHAPE,
        batch_size = BATCH_SIZE,
        class_mode = 'categorical')

test_gen = test_idg.flow_from_directory(
        'intel-image-classification/seg_test/seg_test',
        target_size = IMG_SHAPE,
        batch_size = BATCH_SIZE,
        class_mode='categorical')


# Model
# save https://www.tensorflow.org/guide/keras/save_and_serialize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

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
        steps_per_epoch=NUM_DATA//BATCH_SIZE, 
        validation_data=test_gen,
        validation_steps=NUM_TEST//BATCH_SIZE,
        epochs=50)