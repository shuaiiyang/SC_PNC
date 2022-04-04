# -*- coding: utf-8 -*-
# @Author  : Shuai_Yang
# @Time    : 2022/3/5

import copy
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from Model import ChannelLayerRelay, ChannelLayer, ResidualBlockTx, ResidualBlock, ResidualBlockRx, \
    PeakSignalToNoiseRatio, TxModel, TxRModel, RxModel
from Channel import channel_relay, channel
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签


# 显示图片结果
def disp_result(decode, range_SNR, interval):
    # SNR间隔
    pl_rangeSNR = [x for x in range_SNR if x % interval == 0]
    # 设置图片大小
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    for i in range(len(pl_rangeSNR)):
        # original_A -> 第一行
        p1 = plt.subplot(4, len(pl_rangeSNR), i + 1)
        p1.imshow(decode['original_A'][i].reshape(28, 28), cmap='gray')  # 显示灰度图
        p1.set_title('Original_A', fontproperties="SimHei", fontsize=10)
        # trans_A -> 第二行
        p2 = plt.subplot(4, len(pl_rangeSNR), i + len(pl_rangeSNR) + 1)
        p2.imshow(decode['trans_A'][i].reshape(28, 28), cmap='gray')  # 显示灰度图
        p2.set_title('SNR = ' + str(pl_rangeSNR[i]), fontproperties="SimHei", fontsize=10)
        # original_B -> 第三行
        p3 = plt.subplot(4, len(pl_rangeSNR), i + 2 * len(pl_rangeSNR) + 1)
        p3.imshow(decode['original_B'][i].reshape(28, 28), cmap='gray')  # 显示灰度图
        p3.set_title('Original_B', fontproperties="SimHei", fontsize=10)
        # trans_B -> 第四行
        p4 = plt.subplot(4, len(pl_rangeSNR), i + 3 * len(pl_rangeSNR) + 1)
        p4.imshow(decode['trans_B'][i].reshape(28, 28), cmap='gray')  # 显示灰度图
        p4.set_title('SNR = ' + str(pl_rangeSNR[i]), fontproperties="SimHei", fontsize=10)
    plt.savefig('../Semantic_PNC_test_LN/Results/picture_predict.png', dpi=200)
    plt.close()
    # plt.show()


# 模型的PSNR
def plot_model_performance(psnr, psnr_A, psnr_B, range_SNR, name):
    # PSNR
    plt.title('不同SNR图片PSNR变化')
    plt.plot(range_SNR,
             psnr,
             label='PSNR_Average',
             linewidth='1',  # 粗细
             color='r',  # 颜色
             linestyle='-',  # 线型（linestyle 简写为 ls）
             marker='D'  # 点型（标记marker）
             )
    plt.plot(range_SNR,
             psnr_A,
             label='PSNR:A->B',
             linewidth='1',  # 粗细
             color='g',  # 颜色
             linestyle='-',  # 线型（line_style 简写为 ls）
             marker='.'  # 点型（标记marker）
             )
    plt.plot(range_SNR,
             psnr_B,
             label='PSNR:B->A',
             linewidth='1',  # 粗细
             color='c',  # 颜色
             linestyle='-',  # 线型（line_style 简写为 ls）
             marker='.'  # 点型（标记marker）
             )
    plt.legend(fontsize=8, loc='lower right')
    plt.ylabel('Peak signal-to-noise ratio (dB)')
    plt.xlabel('Eb/N0(dB)')
    plt.ylim([18, 28])
    plt.grid(linestyle='-.')
    file_dir = '../Semantic_PNC_test_LN/Results/' + name + '.png'
    plt.savefig(file_dir, dpi=200)
    plt.close()
    # plt.show()


# 显示图片
def show_images(decode_images, x_test, position_compare):
    """
    plot the images.
    :param decode_images: the images after decoding
    :param x_test: testing data
    :return:
    """
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        ax.imshow(x_test[position_compare + i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        ax.imshow(decode_images[position_compare + i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# Semantic  communication model
def se_model():
    SE_model = keras.models.load_model('../Semantic_PNC_test_LN/Models/3/SE_model.h5',
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
    # SE_model.summary()
    # Tx encoder
    # Node A
    Tx_en_A = keras.Model(inputs=SE_model.get_layer('b_A').input,
                          outputs=SE_model.get_layer('TxModel_A').output)
    # keras.utils.plot_model(Tx_en_A, to_file='../Semantic_PNC_test_LN/Models/Block_Model/Tx_en_A.png', show_shapes=True)
    # Node B
    Tx_en_B = keras.Model(inputs=SE_model.get_layer('b_B').input,
                          outputs=SE_model.get_layer('TxModel_B').output)
    # keras.utils.plot_model(Tx_en_B, to_file='../Semantic_PNC_test_LN/Models/Block_Model/Tx_en_B.png', show_shapes=True)
    # relay
    TxRxR = keras.Model(inputs=SE_model.get_layer('TxRModel').input,
                        outputs=SE_model.get_layer('TxRModel').output)
    # keras.utils.plot_model(TxRxR, to_file='../Semantic_PNC_test_LN/Models/Block_Model/TxRxR.png', show_shapes=True)
    # RX decoder
    # Node A
    Rx_de_B = keras.Model(inputs=SE_model.get_layer('RxModel_A').input,
                          outputs=SE_model.get_layer('RxModel_A').output)
    # keras.utils.plot_model(Rx_de_B, to_file='../Semantic_PNC_test_LN/Models/Block_Model/Rx_de_B.png', show_shapes=True)
    # Node B
    Rx_de_A = keras.Model(inputs=SE_model.get_layer('RxModel_B').input, 
                          outputs=SE_model.get_layer('RxModel_B').output)
    # keras.utils.plot_model(Rx_de_A, to_file='../Semantic_PNC_test_LN/Models/Block_Model/Rx_de_A.png', show_shapes=True)

    return Tx_en_A, Tx_en_B, TxRxR, Rx_de_A, Rx_de_B


# 预测模型
def predict_model():
    # load model
    Tx_en_A, Tx_en_B, TxRxR, Rx_de_A, Rx_de_B = se_model()

    # SNR range
    SNR = range(-10, 26)
    # peak signal-to-noise ratio
    PSNR = []
    PSNR_A = []
    PSNR_B = []
    # 预测结果
    x_predict = {'original_A': [], 'original_B': [], 'trans_A': [], 'trans_B': []}

    # Step1: load data
    _, (x_test, _) = keras.datasets.mnist.load_data()
    # cifar = keras.datasets.cifar100
    # _, (x_test, _) = cifar.load_data(label_mode='coarse')
    # x_test = tf.image.resize(images=x_test, size=(28, 28))
    # x_test = tf.image.rgb_to_grayscale(x_test)
    # x_test = x_test / 255.  # 标准化
    # x_test = np.array(x_test)
    x_test = x_test.astype('float32') / 255  # 标准化
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    for SNR_dB in SNR:
        # 随机对比的图片
        position_compare = random.randint(0, x_test.shape[0] - 1)
        # print('SNR=' + str(SNR_dB))
        print('SNR_dB=' + str(SNR_dB))
        x_test_A = copy.deepcopy(x_test)
        x_test_B = copy.deepcopy(x_test)
        random.shuffle(x_test_A)
        random.shuffle(x_test_B)

        # Step2: Predict
        x_A = Tx_en_A.predict(x=x_test_A)
        x_B = Tx_en_B.predict(x=x_test_B)
        # signal overlapping
        y_R = channel_relay(inputs=[x_A, x_B], 
                            snra_db=SNR_dB,
                            snrb_db=SNR_dB+2,
                            channel_type='AWGN',
                            modulation='QPSK',
                            phase_offsets=0)
        x_R = TxRxR.predict(x=y_R)
        y_A = channel(inputs=x_R, snr_db=SNR_dB, channel_type='AWGN', modulation='QPSK')
        y_B = channel(inputs=x_R, snr_db=SNR_dB+2, channel_type='AWGN', modulation='QPSK')
        # Node A
        b_B = Rx_de_B.predict(x=[x_A, y_A])
        # Node B
        b_A = Rx_de_A.predict(x=[x_B, y_B])

        # Step3: Results
        if SNR_dB % 5 == 0:
            x_predict['original_A'].append(x_test_A[position_compare].copy())
            x_predict['original_B'].append(x_test_B[position_compare].copy())
            x_predict['trans_A'].append(b_A[position_compare].copy())
            x_predict['trans_B'].append(b_B[position_compare].copy())
        psnr_A = tf.image.psnr(b_A, x_test_A, max_val=1.0)  # A->B
        PSNR_A.append(tf.reduce_mean(psnr_A).numpy())
        psnr_B = tf.image.psnr(b_B, x_test_B, max_val=1.0)  # B->A
        PSNR_B.append(tf.reduce_mean(psnr_B).numpy())
        psnr = tf.reduce_mean([psnr_A, psnr_B])
        PSNR.append(psnr.numpy())
        print('PSNR_A->B='+str(tf.reduce_mean(psnr_A).numpy())+'  PSNR_B->A='+str(tf.reduce_mean(psnr_B).numpy()))
    plot_model_performance(PSNR, PSNR_A, PSNR_B, SNR, name='PSNR_SNR')
    disp_result(x_predict, SNR, interval=5)


# 测试模型准确率
def evaluate_model():
    # 加载测试集
    pass


if __name__ == '__main__':
    predict_model()
