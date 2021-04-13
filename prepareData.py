import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import ndimage
import random
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
def deleteElements(list, difference):
    for i in range(difference):
        list.pop(random.randrange(0,len(list)))

def prepare(path):
    apnea_paths = [
        os.path.join(os.getcwd(), path+'apnea/' , x)
        for x in os.listdir(path+'apnea/')
    ]

    nonapnea_paths = [
        os.path.join(os.getcwd(), path+'nonapnea/' , x)
        for x in os.listdir(path+'nonapnea/')
    ]

    print("Apnea samples: " + str(len(apnea_paths)))
    print("Nonapnea Samples: " + str(len(nonapnea_paths)))
    difference = abs(len(apnea_paths)-len(nonapnea_paths))
    if len(apnea_paths)>len(nonapnea_paths):
        deleteElements(apnea_paths,difference)
    else:
        deleteElements(nonapnea_paths,difference)
    apnea = np.array([np.load(path) for path in apnea_paths])
    nonapnea = np.array([np.load(path) for path in nonapnea_paths])
    apnea_labels = np.array([1 for _ in range(len(apnea))])
    nonapnea_labels = np.array([0 for _ in range(len(nonapnea))])

    x = np.concatenate((apnea, nonapnea), axis=0).astype(np.float32)
    y = np.concatenate((apnea_labels, nonapnea_labels), axis=0)
    return x, y


def main():
    x_train, y_train = prepare('/home/zceeyan/project/3Ddataset/normalized/training/')
    x_val, y_val = prepare('/home/zceeyan/project/3Ddataset/normalized/validation/')
    x_test, y_test = prepare('/home/zceeyan/project/3Ddataset/normalized/testing/')

    print(x_train.dtype)
    print(
        "Number of samples in train and validation are %d and %d."
        % (x_train.shape[0], x_val.shape[0])
    )
    train_dataset, validation_dataset = dataLoader(x_train, y_train, x_val, y_val)
    train_dataset.save_dataset('train_dataset')
    np.save('train_dataset', train_dataset)
    np.save('validation_dataset',validation_dataset)

if __name__ == '__main__':
    main()