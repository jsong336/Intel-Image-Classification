# Model
IMG_SHAPE = (128, 128)
BATCH_SIZE = 32

# Train
NUM_DATA = 2000 
NUM_TEST = 200
EPOCHS = 1

# Datasets
PATH = 'intel-image-classification/'
PATH_TRAIN = PATH + 'seg_train/seg_train'
PATH_TEST = PATH + 'seg_test/seg_test'

MAP = [
    'building',
    'forest',
    'glacier',
    'mountain',
    'sea',
    'street'
    ]

"""
Ubuntu NVIDA => sad life
NUM_DATA = 14034 
NUM_TEST = 3000
EPOCHS = 50
"""