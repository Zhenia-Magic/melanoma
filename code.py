#import needed packages and libraries
!pip install -q efficientnet
!export TF_ENABLE_AUTO_MIXED_PRECISION=1

import numpy as np
import pandas as pd 
import os
import re
from kaggle_datasets import KaggleDatasets
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import optimizers
import efficientnet.tfkeras as efn
from sklearn.model_selection import KFold

# Detect TPU and configure the system appropriately
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

#get dataset from Kaggle to use it with TPU
DATASET = '512x512-melanoma-tfrecords-70k-images'
GCS_PATH = KaggleDatasets().get_gcs_path(DATASET)

#set required parameters
SEED = 42
SIZE = [512,512]
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
EPOCHS = 10
TTA = 4
LR = 0.00004
WARMUP = 5
LABEL_SMOOTHING = 0.05
AUTO = tf.data.experimental.AUTOTUNE

#fix random seed for reproducibility
def seed_everything(SEED):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

seed_everything(SEED)

#create a function to augment the image with different augmentations such as rotations,
#flips, changes in hues, saturation, contrast, and brightness to enhance the generalization of the model
def data_augment(image, label=None, seed=SEED):
    image = tf.image.rot90(image,k=np.random.randint(4))
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_con(Deotte, 2020)ΩdWQtrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)
    if label is None:
        return image
    else:
        return image, label
        
#create a function  to decode image from jpeg and reshape it to the right size
def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3) 
    image = tf.cast(image, tf.float32)/255.0
    image = tf.reshape(image, [*SIZE, 3])
    return image

# create a function to read the image and target (train/valid data) from tfrecord    
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),  }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['target'], tf.int32)
    return image, label 

# create a function to read the image and image name (test data) from tfrecord
def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string), }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    image_name = example['image_name']
    return image, image_name

# create a function to read full dataset from tfrecords
def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = (tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 
              .with_options(ignore_order)
              .map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO))
    return dataset
    

#create function to load training dataset, augment images in it, shuffle, and batch it        
def get_training_dataset(filenames, labeled = True, ordered = False):
    dataset = load_dataset(filenames, labeled = labeled, ordered = ordered)
    dataset = dataset.map(data_augment, num_parallel_calls = AUTO)
    # the training dataset must repeat for several epochs
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTO)
    return dataset

#create function to load validation dataset, augment images in it, shuffle, and batch it
def get_validation_dataset(filenames, labeled = True, ordered = True):
    dataset = load_dataset(filenames, labeled = labeled, ordered = ordered)
    dataset = dataset.batch(BATCH_SIZE)
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTO) 
    return dataset

#create function to load test dataset and batch it
def get_test_dataset(filenames, labeled = False, ordered = True):
    dataset = load_dataset(filenames, labeled = labeled, ordered = ordered)
    dataset = dataset.map(data_augment, num_parallel_calls = AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTO) 
    return dataset
    
#read train and test filenames from the data directory
train_filenames = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')
test_filenames = tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')

#display random picture with its augmentations
def plot_transform(num_images):
    plt.figure(figsize=(30,10))
    x = load_dataset(train_filenames, labeled=False)
    image,_ = iter(x).next()
    for i in range(1,num_images+1):
        plt.subplot(1,num_images+1,i)
        plt.axis('off')
        image = data_augment(image=image)
        plt.imshow(image)
        
plot_transform(7)

#create a function to define a model that will be trained
#here, I use transfer learning with the Efficient net B7 model
#which is one of the most advanced model for image classification nowadays

def get_model():
    with strategy.scope():

        model = tf.keras.Sequential([
            efn.EfficientNetB7(input_shape=(*SIZE, 3),weights='imagenet',pooling='avg',include_top=False),
            Dense(1, activation='sigmoid')
        ])
    
        model.compile(
            optimizer='adam',
            loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = LABEL_SMOOTHING),
            metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
    return model
    
#create a function that defines the learning rate schedule
#using cosine schedule with warmups
def get_cosine_schedule_with_warmup(lr,num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr
        progress = float(epoch - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

lr_schedule= get_cosine_schedule_with_warmup(lr=LR,num_warmup_steps=WARMUP,num_training_steps=EPOCHS)

#create function to count data items in the filenames directory
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

#count the number of train, valid, and test files as well as steps per epochs
NUM_TRAINING_IMAGES = int(count_data_items(train_filenames) * 0.8)
NUM_VALIDATION_IMAGES = int(count_data_items(train_filenames) * 0.2)
NUM_TEST_IMAGES = count_data_items(test_filenames)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

#define the function that will perform training with the k-folds cross entropy
def train(folds = 5):
    
    models = []
    
    # seed everything
    seed_everything(SEED)

    kfold = KFold(folds, shuffle = True, random_state = SEED)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train_filenames)):
        print('\n')
        print('-'*50)
        print(f'Training fold {fold + 1}')
        train_dataset = get_training_dataset([train_filenames[x] for x in trn_ind], labeled = True, ordered = False)
        val_dataset = get_validation_dataset([train_filenames[x] for x in val_ind], labeled = True, ordered = True)
        K.clear_session()
        model = get_model()
        # using early stopping using val loss
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_auc', mode = 'max', patience = 5, 
                                                      verbose = 1, min_delta = 0.0001, restore_best_weights = True)
        history = model.fit(train_dataset, 
                            steps_per_epoch = STEPS_PER_EPOCH,
                            epochs = EPOCHS,
                            callbacks = [early_stopping, lr_schedule],
                            validation_data = val_dataset,
                            verbose = 1)
        models.append(model)
    
    print('\n')
    print('-'*50)
    submission_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
    print('Computing predictions...')
    #using TTA to predict the test dataset predictions
    for i in range(TTA):
        test_ds = get_test_dataset(test_filenames, labeled = False, ordered = True)
        test_images_ds = test_ds.map(lambda image, image_name: image)
        probabilities = np.average([np.concatenate(models[i].predict(test_images_ds)) for i in range(folds)], axis = 0)
        test_ids_ds = test_ds.map(lambda image, image_name: image_name).unbatch()
        # all in one batch
        test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')
        pred_df = pd.DataFrame({'image_name': test_ids, 'target': probabilities})
        temp = submission_df.copy()   
        del temp['target']  
        submission_df['target'] += temp.merge(pred_df,on="image_name")['target']/TTA
    print('Generating submission.csv file...')
    submission_df.to_csv('submission.csv', index=False)
    return submission_df
    
    train(5)
