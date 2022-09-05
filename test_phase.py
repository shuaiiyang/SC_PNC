# -*- coding: utf-8 -*-
# @Author  : Shuai_Yang
# @Time    : 2022/3/5

import os
import copy
import glob
import keras
import random
import scipy.io as scio
import tensorflow as tf
import matplotlib.pyplot as plt

from model import  semantic_twrc as creat_model
from channel import channel_relay, channel
from train import parse_args

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


def data_loader():
    """
    load datasets
    """
    # Step1: load data mnist
    mnist = tf.keras.datasets.mnist
    _, (x_test, _) = mnist.load_data()
    # Step2: normalize
    x_test_ = x_test.astype('float32') / 255.
    x_test_ = x_test_.reshape(x_test.shape[0], 28, 28, 1)
    x_test_A = copy.deepcopy(x_test_[:5000])
    x_test_B = copy.deepcopy(x_test_[:5000])
    # Step3: shuffle
    random.shuffle(x_test_A)
    random.shuffle(x_test_B)

    return (x_test_A, x_test_B)


def disp_result(decode, range_phase, interval):
    """
    visualization
    :param decode: image data
    :param range_phase: Test the range of Relative Phase Offset
    :param interval: Relative Phase Offset value interval
    :return:
    """
    # phase interval
    pl_rangePhase = [x for x in range_phase if x % interval == 0]
    # image size
    plt.figure(figsize=(14, 8))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    for i in range(len(pl_rangePhase)):
        # original_A -> first row
        p1 = plt.subplot(4, len(pl_rangePhase), i + 1)
        p1.imshow(decode['original_A'][i].reshape(28, 28), cmap='gray')  # disp
        p1.set_title('Original_A', fontproperties="SimHei", fontsize=10)
        # trans_A -> second row
        p2 = plt.subplot(4, len(pl_rangePhase), i + len(pl_rangePhase) + 1)
        p2.imshow(decode['trans_A'][i].reshape(28, 28), cmap='gray')  # disp
        p2.set_title('Phase_A = ' + str(pl_rangePhase[i]), fontproperties="SimHei", fontsize=10)
        # original_B -> third row
        p3 = plt.subplot(4, len(pl_rangePhase), i + 2 * len(pl_rangePhase) + 1)
        p3.imshow(decode['original_B'][i].reshape(28, 28), cmap='gray')  # disp
        p3.set_title('Original_B', fontproperties="SimHei", fontsize=10)
        # trans_B -> fourth row
        p4 = plt.subplot(4, len(pl_rangePhase), i + 3 * len(pl_rangePhase) + 1)
        p4.imshow(decode['trans_B'][i].reshape(28, 28), cmap='gray')  # disp
        p4.set_title('Phase_B = ' + str(pl_rangePhase[i]), fontproperties="SimHei", fontsize=10)
    plt.savefig('SC_PNC/results/picture_predict_phase.png', dpi=200)
    plt.close()
    # plt.show()


def plot_model_performance(psnr, psnr_A, psnr_B, range_phase, name):
    """
    plot PSNR performance
    :param psnr_A: PSNR performance sent by node A to node B
    :param psnr_B: PSNR performance sent by node B to node A
    :param psnr: the average of psnr_A and psnr_B
    :param range_phase: Test the range of Relative Phase Offset
    :param name: name of save image
    :return:
    """
    plt.title('Effect of Relative Phase Offset on PSNR')
    plt.plot(range_phase,
             psnr,
             label='PSNR_Average',
             linewidth='1',
             color='r',
             linestyle='-',
             marker='D'
             )
    plt.plot(range_phase,
             psnr_A,
             label='PSNR_A',
             linewidth='1',
             color='g',
             linestyle='-',
             marker='.'
             )
    plt.plot(range_phase,
             psnr_B,
             label='PSNR_B',
             linewidth='1',
             color='c',
             linestyle='-',
             marker='.'
             )
    plt.legend(fontsize=8, loc='upper right')
    plt.ylabel('Peak signal-to-noise ratio (dB)')
    plt.xlabel('Phase offsets')
    plt.ylim([15, 30])
    plt.grid(linestyle='-.')
    file_dir = 'SC_PNC/results/' + name + '.png'
    plt.savefig(file_dir, dpi=200)
    plt.close()
    # plt.show()



def sc_model():
    """
    semantic communication empowered physical-layer network coding
    """
    args = parse_args()
    SC_PNC = creat_model(args)
    SC_PNC.build(input_shape=[(1,28,28,1),(1,28,28,1)])
    SC_PNC.summary()
    # load weight for MAE SC
    weights_path = 'SC_PNC/save_weights/model.ckpt'
    assert len(glob.glob(weights_path+'*')), "cannot find {}".format(weights_path)
    SC_PNC.load_weights(weights_path)

    # Tx -> Node A
    Tx_en_A = keras.Sequential([SC_PNC.get_layer('SemanticEncoder_A'), SC_PNC.get_layer('ChannelEncoder_A')])
    # Tx -> Node B
    Tx_en_B = keras.Sequential([SC_PNC.get_layer('SemanticEncoder_B'), SC_PNC.get_layer('ChannelEncoder_B')])
    # relay
    TxRxR = keras.Sequential([SC_PNC.get_layer('Semantic_PNC')])
    # Rx -> Node A
    Rx_de_B = keras.Sequential([SC_PNC.get_layer('ChannelDecoder_A'), SC_PNC.get_layer('SemanticDecoder_A')])
    # Rx -> Node B
    Rx_de_A = keras.Sequential([SC_PNC.get_layer('ChannelDecoder_B'), SC_PNC.get_layer('SemanticDecoder_B')])

    return Tx_en_A, Tx_en_B, TxRxR, Rx_de_A, Rx_de_B


def test_model():
    if not os.path.exists("SC_PNC/results"):
        # Create a folder to save results
        os.makedirs("SC_PNC/results")

    batch_size=128
    # load model
    Tx_en_A, Tx_en_B, TxRxR, Rx_de_A, Rx_de_B = sc_model()
    (x_test_1, x_test_2) = data_loader()
    # SNR range
    # SNR = range(-10, 26)
    SNR_dB = 1
    # phase_offsets
    phase_offsets = range(0, 91, 5)
    # peak signal-to-noise ratio
    PSNR = []
    PSNR_A = []
    PSNR_B = []
    # save image to visualization
    x_predict = {'original_A': [], 'original_B': [], 'trans_A': [], 'trans_B': []}
    for phase in phase_offsets:
        # randomly choose a location for comparison
        position_compare = random.randint(0, x_test_1.shape[0] - 1)
        print('phase=' + str(phase))

        # Predict
        x_A = Tx_en_A.predict(x=x_test_1, batch_size=batch_size)
        x_B = Tx_en_B.predict(x=x_test_2, batch_size=batch_size)
        # signal overlapping
        y_R = channel_relay(inputs=[x_A, x_B],
                            snra_db=SNR_dB,
                            snrb_db=SNR_dB,
                            channel_type='AWGN',
                            phase_offsets=phase)
        x_R = TxRxR.predict(x=y_R, batch_size=batch_size)
        y_A = channel(inputs=x_R, snr_db=SNR_dB, channel_type='AWGN', modulation='QPSK')
        y_B = channel(inputs=x_R, snr_db=SNR_dB, channel_type='AWGN', modulation='QPSK')
        # Node A
        b_B = Rx_de_B.predict(x=[y_A, x_A], batch_size=batch_size)
        # Node B
        b_A = Rx_de_A.predict(x=[y_B, x_B], batch_size=batch_size)

        # save results
        if phase % 10 == 0:
            x_predict['original_A'].append(x_test_1[position_compare].copy())
            x_predict['original_B'].append(x_test_2[position_compare].copy())
            x_predict['trans_A'].append(b_A[position_compare].copy())
            x_predict['trans_B'].append(b_B[position_compare].copy())
        psnr_A = tf.image.psnr(b_A, x_test_1, max_val=1.0)
        PSNR_A.append(tf.reduce_mean(psnr_A).numpy())
        psnr_B = tf.image.psnr(b_B, x_test_2, max_val=1.0)
        PSNR_B.append(tf.reduce_mean(psnr_B).numpy())
        psnr = tf.reduce_mean([psnr_A, psnr_B])
        PSNR.append(psnr.numpy())
        print('PSNR_A->B=' + str(tf.reduce_mean(psnr_A).numpy()) + '  PSNR_B->A=' + str(tf.reduce_mean(psnr_B).numpy()))
    plot_model_performance(PSNR, PSNR_A, PSNR_B, phase_offsets, name='PSNR_phase')
    disp_result(x_predict, phase_offsets, interval=10)
    # save data
    # if not os.path.exists("SC_PNC/data"):
    #     # Create a folder to save results
    #     os.makedirs("SC_PNC/data")
    # dataNew = 'SC_PNC/data/Deep_5dot5_degree_0_90.mat'
    # scio.savemat(dataNew, {'phase_offsets': list(phase_offsets), 'PSNR_A_B': PSNR_A, 'PSNR_B_A': PSNR_B})


if __name__ == '__main__':
    test_model()
