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

import os

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
    cifar = keras.datasets.cifar100
    (x_train, _), (x_test, _) = cifar.load_data(label_mode='coarse')
    x_train_ = []
    for i in range(0, len(x_train), 1000):
        x_train_.append(tf.image.rgb_to_grayscale(x_train[i:i + 1000, :, :, :]))
    x_test_ = []
    for i in range(0, len(x_test), 1000):
        x_test_.append(tf.image.rgb_to_grayscale(x_test[i:i + 1000, :, :, :]))
    x_train_ = np.reshape(np.array(x_train_), (x_train.shape[0], 32, 32, 1))
    x_test_ = np.reshape(np.array(x_test_), (x_test.shape[0], 32, 32, 1))

    # Step2: normalize
    x_train_ = x_train_.astype('float32') / 255.
    x_test_ = x_test_.astype('float32') / 255.
    # x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    # x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train_A = copy.deepcopy(x_train_)
    x_train_B = copy.deepcopy(x_train_)
    x_test_A = copy.deepcopy(x_test_)
    x_test_B = copy.deepcopy(x_test_)
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
    # train_ds, val_ds = generate_ds(data_root='',
    #                                train_im_height=32,
    #                                train_im_width=32,
    #                                batch_size=50,
    #                                cache_data=False)

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
    epochs = 3
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
