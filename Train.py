# -*- coding: utf-8 -*-
# @Author  : Shuai_Yang
# @Time    : 2022/3/23
"""
train DNN-based PNC
"""



import copy
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from Model import ChannelLayerRelay, ChannelLayer, ResidualBlockTx, ResidualBlock, ResidualBlockRx, \
    PeakSignalToNoiseRatio, TxModel, TxRModel, RxModel
from Model import show_images, PNC_Model

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定显卡
# config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 开启软放置，OP放到CPU上了，为啥源码默认值是7？
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 1  # 进程最多采用30%显存，默认是1，完整使用
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# 加载图片数据
def data_loader():
    # Step1: load data cifar100
    # mnist = keras.datasets.mnist
    # (x_train, _), (x_test, _) = mnist.load_data()
    cifar = keras.datasets.cifar100
    (x_train, _), (x_test, _) = cifar.load_data(label_mode='coarse')
    # x_train = tf.image.resize(images=x_train, size=(28, 28))
    # x_test = tf.image.resize(images=x_test, size=(28, 28))
    x_train = x_train[0:1000, :, :, :]
    x_train = tf.image.rgb_to_grayscale(x_train)
    x_test = tf.image.rgb_to_grayscale(x_test[0:1000, :, :, :])
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # Step2: normalize
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    # x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train_A = copy.deepcopy(x_train)
    x_train_B = copy.deepcopy(x_train)
    x_test_A = copy.deepcopy(x_test)
    x_test_B = copy.deepcopy(x_test)
    # show_images(x_train_A, x_train_B)
    # show_images(x_test_A, x_test_B)
    random.shuffle(x_train_A)
    random.shuffle(x_train_B)
    random.shuffle(x_test_A)
    random.shuffle(x_test_B)
    # show_images(x_train_A, x_train_B)
    # show_images(x_test_A, x_test_B)
    return (x_train_A, x_train_B), (x_test_A, x_test_B)


# main
def main():
    # load data
    (x_train_A, x_train_B), (x_test_A, x_test_B) = data_loader()
    # load model
    # pnc_model = PNC_Model(input_shape=(32, 32, 1))
    pnc_model = keras.models.load_model('../SE_PNC/Models/SE_model_cifar.h5',
                                       {'ChannelLayerRelay': ChannelLayerRelay,
                                        'ChannelLayer': ChannelLayer,
                                        'ResidualBlockTx': ResidualBlockTx,
                                        'ResidualBlock': ResidualBlock,
                                        'ResidualBlockRx': ResidualBlockRx,
                                        'PeakSignalToNoiseRatio': PeakSignalToNoiseRatio,
                                        'TxModel': TxModel,
                                        'TxRModel': TxRModel,
                                        'RxModel': RxModel
                                        })
    pnc_model.summary()

    # Step3: train
    epochs = 5
    # save model graph
    keras.utils.plot_model(pnc_model, to_file='../SE_PNC/Models/SE_model_cifar.png', show_shapes=True)
    pnc_model.fit(x=[x_train_A, x_train_B],
                  y=[x_train_A, x_train_B],
                  batch_size=100,
                  epochs=epochs,
                  validation_data=([x_test_A, x_test_B], [x_test_A, x_test_B]),
                  shuffle=True
                  )
    pnc_model.save("../SE_PNC/Models/SE_model_cifar.h5")


if __name__ == '__main__':
    main()

