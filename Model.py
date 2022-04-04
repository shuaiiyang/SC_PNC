# -*- coding: utf-8 -*-
# @Author  : Shuai_Yang
# @Time    : 2022/2/25
"""
DNN-based PNC
"""

import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras


# Relay 信道层
class ChannelLayerRelay(keras.layers.Layer):
    def __init__(self, snr_db, channel_type, modulation='BPSK', phase_offsets=0, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.snr_db = snr_db
        self.channel_type = channel_type
        self.modulation = modulation
        self.phase_offsets = phase_offsets

    def call(self, inputs, *args, **kwargs):
        Eb = 1 / 2
        # SNR(dB) = Eb/N0(dB) + 10log(k)
        EbN0 = 10 ** (self.snr_db / 10)
        N0 = Eb / EbN0
        sigma = math.sqrt(N0 / 2)
        std = tf.constant(value=sigma, dtype=tf.float32)
        inputs_real_A = inputs[0][:, 0:64, :]
        inputs_imag_A = inputs[0][:, 64:128, :]
        inputs_real_B = inputs[1][:, 0:64, :]
        inputs_imag_B = inputs[1][:, 64:128, :]
        inputs_complex_A = tf.complex(real=inputs_real_A, imag=inputs_imag_A)
        inputs_complex_B = tf.complex(real=inputs_real_B, imag=inputs_imag_B)
        # AWGN channel
        if self.channel_type == 'AWGN':
            phase_offsetsA = tf.random.uniform(shape=(1,), minval=0, maxval=90)
            hA_complex = tf.exp(tf.complex(real=0., imag=math.pi * phase_offsetsA / 180))
            phase_offsetsB = tf.random.uniform(shape=(1,), minval=0, maxval=90)
            hB_complex = tf.exp(tf.complex(real=0., imag=math.pi * phase_offsetsB / 180))
        # Rayleigh channel
        elif self.channel_type == 'Rayleigh':
            print('修改！')
        # noise
        n_real = tf.random.normal(shape=tf.shape(inputs_complex_A), mean=0.0, stddev=std, dtype=tf.float32)
        n_imag = tf.random.normal(shape=tf.shape(inputs_complex_A), mean=0.0, stddev=std, dtype=tf.float32)
        noise = tf.complex(real=n_real, imag=n_imag)
        # received signal y
        hAx = tf.multiply(hA_complex, inputs_complex_A)
        hBx = tf.multiply(hB_complex, inputs_complex_B)
        hx = tf.add(hAx, hBx)
        y_complex = tf.add(hx, noise)
        # # perfect channel estimation
        # y_complex = tf.divide(r_complex, h_complex)
        # reshape
        y_real = tf.math.real(y_complex)
        y_imag = tf.math.imag(y_complex)
        # hA_real = tf.math.real(hA_complex)
        # hA_imag = tf.math.imag(hA_complex)
        # hB_real = tf.math.real(hB_complex)
        # hB_imag = tf.math.imag(hB_complex)
        # output = tf.concat([y_real, hA_real[:, 0:1, :], hB_real[:, 0:1, :], y_imag, hA_imag[:, 0:1, :], hB_imag[:, 0:1, :]], axis=1)
        output = tf.concat([y_real, y_imag], axis=1)

        return output

    def get_config(self):
        base_config = super().get_config()
        base_config['snr_db'] = self.snr_db
        base_config['channel_type'] = self.channel_type
        base_config['modulation'] = self.modulation
        base_config['phase_offsets'] = self.phase_offsets

        return base_config


# 信道层
class ChannelLayer(keras.layers.Layer):
    def __init__(self, snr_db, channel_type, modulation='BPSK', name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.snr_db = snr_db
        self.channel_type = channel_type
        self.modulation = modulation

    def call(self, inputs, *args, **kwargs):
        Eb = 1 / 2
        # SNR(dB) = Eb/N0(dB) + 10log(k)
        EbN0 = 10 ** (self.snr_db / 10)
        N0 = Eb / EbN0
        sigma = math.sqrt(N0 / 2)
        std = tf.constant(value=sigma, dtype=tf.float32)
        inputs_real = inputs[:, 0:64, :]
        inputs_imag = inputs[:, 64:128, :]
        inputs_complex = tf.complex(real=inputs_real, imag=inputs_imag)
        # AWGN channel
        if self.channel_type == 'AWGN':
            h_complex = tf.exp(tf.complex(real=0., imag=math.pi * 0 / 180))
        # Rayleigh channel
        elif self.channel_type == 'Rayleigh':
            h_real = tf.divide(
                tf.random.normal(shape=tf.shape(inputs_complex), mean=0.0, stddev=1.0, dtype=tf.float32),
                tf.sqrt(2.))
            h_imag = tf.divide(
                tf.random.normal(shape=tf.shape(inputs_complex), mean=0.0, stddev=1.0, dtype=tf.float32),
                tf.sqrt(2.))
            h_complex = tf.complex(real=h_real, imag=h_imag)
        # noise
        n_real = tf.random.normal(shape=tf.shape(inputs_complex), mean=0.0, stddev=std, dtype=tf.float32)
        n_imag = tf.random.normal(shape=tf.shape(inputs_complex), mean=0.0, stddev=std, dtype=tf.float32)
        noise = tf.complex(real=n_real, imag=n_imag)
        # received signal y
        hx = tf.multiply(h_complex, inputs_complex)
        y_complex = tf.add(hx, noise)
        # # perfect channel estimation
        # y_complex = tf.divide(r_complex, h_complex)
        # reshape
        y_real = tf.math.real(y_complex)
        y_imag = tf.math.imag(y_complex)
        output = tf.concat([y_real, y_imag], axis=1)

        return output

    def get_config(self):
        base_config = super().get_config()
        base_config['snr_db'] = self.snr_db
        base_config['channel_type'] = self.channel_type
        base_config['modulation'] = self.modulation

        return base_config


# 残差Block_Tx
class ResidualBlockTx(keras.layers.Layer):

    def __init__(self, out_channel, strides=1, downsample=False, name=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.out_channel = out_channel
        self.strides = strides
        self.downsample = downsample
        if self.downsample:
            self.Conv2D_0 = keras.layers.Conv2D(filters=self.out_channel, kernel_size=1, strides=self.strides,
                                                padding='same', activation=None)
            self.BatchNorm_0 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.Conv2D_1 = keras.layers.Conv2D(filters=self.out_channel, kernel_size=3, strides=self.strides,
                                            padding='same', activation=None)
        self.Conv2D_2 = keras.layers.Conv2D(filters=self.out_channel, kernel_size=3, strides=1, padding='same',
                                            activation=None)
        self.BatchNorm_1 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.BatchNorm_2 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.downsample = downsample
        self.ELU = keras.layers.ELU()
        self.Add = keras.layers.Add()

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        if self.downsample:
            residual = self.Conv2D_0(residual)
            residual = self.BatchNorm_0(residual)
        x = self.Conv2D_1(inputs)
        x = self.BatchNorm_1(x)
        x = self.ELU(x)
        x = self.Conv2D_2(x)
        x = self.BatchNorm_2(x)
        x = self.Add([x, residual])
        x = self.ELU(x)

        return x

    def get_config(self):
        base_config = super().get_config()
        base_config['out_channel'] = self.out_channel
        base_config['strides'] = self.strides
        base_config['downsample'] = self.downsample

        return base_config


# 残差Block_R
class ResidualBlock(keras.layers.Layer):

    def __init__(self, out_channel, strides=1, downsample=False, name=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.out_channel = out_channel
        self.strides = strides
        self.downsample = downsample
        if self.downsample:
            self.Conv1D_0 = keras.layers.Conv1D(filters=self.out_channel, kernel_size=1, strides=self.strides,
                                                padding='same', activation=None)
            self.BatchNorm_0 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.Conv1D_1 = keras.layers.Conv1D(filters=self.out_channel, kernel_size=3, strides=self.strides,
                                            padding='same', activation=None)
        self.Conv1D_2 = keras.layers.Conv1D(filters=self.out_channel, kernel_size=3, strides=1, padding='same',
                                            activation=None)
        self.BatchNorm_1 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.BatchNorm_2 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.ELU = keras.layers.ELU()
        self.Add = keras.layers.Add()

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        if self.downsample:
            residual = self.Conv1D_0(residual)
            residual = self.BatchNorm_0(residual)
        x = self.Conv1D_1(inputs)
        x = self.BatchNorm_1(x)
        x = self.ELU(x)
        x = self.Conv1D_2(x)
        x = self.BatchNorm_2(x)
        x = self.Add([x, residual])
        x = self.ELU(x)

        return x

    def get_config(self):
        base_config = super().get_config()
        base_config['out_channel'] = self.out_channel
        base_config['strides'] = self.strides
        base_config['downsample'] = self.downsample

        return base_config


# 残差Block_Rx
class ResidualBlockRx(keras.layers.Layer):

    def __init__(self, out_channel, strides=1, downsample=False, name=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.out_channel = out_channel
        self.strides = strides
        self.downsample = downsample
        if self.downsample:
            self.Conv2DTrans_0 = keras.layers.Conv2DTranspose(filters=self.out_channel, kernel_size=1,
                                                              strides=self.strides, padding='same', activation=None)
            self.BatchNorm_0 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.Conv2DTrans_1 = keras.layers.Conv2DTranspose(filters=self.out_channel, kernel_size=3, strides=self.strides,
                                                          padding='same', activation=None)
        self.Conv2DTrans_2 = keras.layers.Conv2DTranspose(filters=self.out_channel, kernel_size=3, strides=1,
                                                          padding='same', activation=None)
        self.BatchNorm_1 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.BatchNorm_2 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.ELU = keras.layers.ELU()
        self.Add = keras.layers.Add()

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        if self.downsample:
            residual = self.Conv2DTrans_0(residual)
            residual = self.BatchNorm_0(residual)
        x = self.Conv2DTrans_1(inputs)
        x = self.BatchNorm_1(x)
        x = self.ELU(x)
        x = self.Conv2DTrans_2(x)
        x = self.BatchNorm_2(x)
        x = self.Add([x, residual])
        x = self.ELU(x)

        return x

    def get_config(self):
        base_config = super().get_config()
        base_config['out_channel'] = self.out_channel
        base_config['strides'] = self.strides
        base_config['downsample'] = self.downsample

        return base_config


# 发送 Model
class TxModel(keras.layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(TxModel, self).__init__(name=name, **kwargs)
        self.ResidualBlock_1 = ResidualBlockTx(out_channel=8, strides=2, downsample=True)
        self.ResidualBlock_2 = ResidualBlockTx(out_channel=8, strides=1, downsample=False)
        self.ResidualBlock_3 = ResidualBlockTx(out_channel=16, strides=2, downsample=True)
        self.ResidualBlock_4 = ResidualBlockTx(out_channel=16, strides=1, downsample=False)
        self.ResidualBlock_5 = ResidualBlockTx(out_channel=32, strides=2, downsample=True)
        self.ResidualBlock_6 = ResidualBlockTx(out_channel=32, strides=1, downsample=False)
        self.Flatten = keras.layers.Flatten()
        self.Dense = keras.layers.Dense(units=128, activation=None)
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=1e-5)
        self.Reshape = keras.layers.Reshape((128, 1))

    def call(self, inputs, *args, **kwargs):
        x = self.ResidualBlock_1(inputs)
        x = self.ResidualBlock_2(x)
        x = self.ResidualBlock_3(x)
        x = self.ResidualBlock_4(x)
        x = self.ResidualBlock_5(x)
        x = self.ResidualBlock_6(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        x = self.LayerNorm(x)

        return self.Reshape(x)


# 中继 Model
class TxRModel(keras.layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(TxRModel, self).__init__(name=name, **kwargs)
        # self.Dense_1 = keras.layers.Dense(units=128, activation=None)
        # self.Reshape_1 = keras.layers.Reshape((128, 1))
        self.ResidualBlock_1 = ResidualBlock(out_channel=32, strides=1, downsample=True)
        self.ResidualBlock_2 = ResidualBlock(out_channel=32, strides=1, downsample=False)
        self.Flatten = keras.layers.Flatten()
        self.Dense = keras.layers.Dense(units=128, activation=None)
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=1e-5)
        self.Reshape = keras.layers.Reshape((128, 1))

    def call(self, inputs, *args, **kwargs):
        # x = self.Flatten(inputs)
        # x = self.Dense_1(x)
        # x = self.Reshape_1(x)
        x = self.ResidualBlock_1(inputs)
        x = self.ResidualBlock_2(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        x = self.LayerNorm(x)

        return self.Reshape(x)


# 接收 Model
class RxModel(keras.layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(RxModel, self).__init__(name=name, **kwargs)
        self.Concatenate = keras.layers.Concatenate(axis=1)
        self.Dense_1 = keras.layers.Dense(units=4 * 4 * 32, activation=None)
        self.Reshape_1 = keras.layers.Reshape((4, 4, 32))
        self.ResidualBlock_1 = ResidualBlockRx(out_channel=32, strides=1, downsample=False)
        self.ResidualBlock_2 = ResidualBlockRx(out_channel=16, strides=2, downsample=True)
        self.ResidualBlock_3 = ResidualBlockRx(out_channel=16, strides=1, downsample=False)
        self.ResidualBlock_4 = ResidualBlockRx(out_channel=8, strides=2, downsample=True)
        self.ResidualBlock_5 = ResidualBlockRx(out_channel=8, strides=1, downsample=False)
        self.ResidualBlock_6 = ResidualBlockRx(out_channel=4, strides=2, downsample=True)
        self.ResidualBlock_7 = ResidualBlockRx(out_channel=2, strides=1, downsample=True)
        self.Flatten = keras.layers.Flatten()
        self.Dense_2 = keras.layers.Dense(units=32 * 32, activation=None)
        self.Sigmoid = keras.layers.Activation('sigmoid')
        self.Reshape_2 = keras.layers.Reshape((32, 32, 1))

    def call(self, inputs, *args, **kwargs):
        Concat = self.Concatenate(inputs)
        x = self.Flatten(Concat)
        x = self.Dense_1(x)
        x = self.Reshape_1(x)
        x = self.ResidualBlock_1(x)
        x = self.ResidualBlock_2(x)
        x = self.ResidualBlock_3(x)
        x = self.ResidualBlock_4(x)
        x = self.ResidualBlock_5(x)
        x = self.ResidualBlock_6(x)
        x = self.ResidualBlock_7(x)
        x = self.Flatten(x)
        x = self.Dense_2(x)
        x = self.Sigmoid(x)

        return self.Reshape_2(x)


# Custom_metric -> PSNR
class PeakSignalToNoiseRatio(keras.metrics.Metric):
    def __init__(self, name="peak_signal-to-noise_ratio", **kwargs):
        super(PeakSignalToNoiseRatio, self).__init__(name=name, **kwargs)
        self.PSNR = self.add_weight(name="PSNR", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        PSNR = tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            PSNR = tf.multiply(PSNR, sample_weight)
        self.PSNR.assign(PSNR)

    def result(self):
        return self.PSNR

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.PSNR.assign(0.0)


# PNC Model
def PNC_Model(input_shape):
    # Node A
    b_A = keras.layers.Input(shape=input_shape, name='b_A')
    # Node B
    b_B = keras.layers.Input(shape=input_shape, name='b_B')
    x_A = TxModel(name='TxModel_A')(b_A)
    x_B = TxModel(name='TxModel_B')(b_B)
    y_R = ChannelLayerRelay(snr_db=7, channel_type='AWGN', modulation='QPSK', phase_offsets=0)([x_A, x_B])
    x_R = TxRModel(name='TxRModel')(y_R)
    y_A = ChannelLayer(snr_db=7, channel_type='AWGN', modulation='QPSK', name='ChannelLayer_A')(x_R)
    y_B = ChannelLayer(snr_db=7, channel_type='AWGN', modulation='QPSK', name='ChannelLayer_B')(x_R)
    # Node A
    b_B_output = RxModel(name='RxModel_A')([x_A, y_A])
    # Node B
    b_A_output = RxModel(name='RxModel_B')([x_B, y_B])

    model = keras.Model(inputs=[b_A, b_B], outputs=[b_A_output, b_B_output])
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss = keras.losses.MeanSquaredError()
    metrics = PeakSignalToNoiseRatio()
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[1., 1.], metrics=metrics)

    return model


# show images
def show_images(decode_images, x_test):
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
        ax.imshow(x_test[i].reshape(28, 28, 1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        ax.imshow(decode_images[i].reshape(28, 28, 1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
