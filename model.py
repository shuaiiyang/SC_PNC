# -*- coding: utf-8 -*-
# @Author  : Shuai_Yang
# @Time    : 2022/2/25
"""
SC PNC TWRC
"""
import math
from keras import layers, Model
import tensorflow as tf


# node -> relay channel
class ChannelLayerRelay(layers.Layer):
    def __init__(self, snr_db, channel_type: str='AWGN', phase_offsets: int=0, name: str=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.snr_db = snr_db
        self.channel_type = channel_type
        self.phase_offsets = phase_offsets

    def call(self, inputs, *args, **kwargs):
        # noise std
        Eb = 1
        EbN0 = 10 ** (self.snr_db / 10)
        N0 = Eb / EbN0
        sigma = math.sqrt(N0 / 2)
        std = tf.constant(value=sigma, dtype=tf.float32)
        # signal
        inputs_real_A = inputs[0][:, :, 0]
        inputs_imag_A = inputs[0][:, :, 1]
        inputs_real_B = inputs[1][:, :, 0]
        inputs_imag_B = inputs[1][:, :, 1]
        inputs_complex_A = tf.complex(real=inputs_real_A, imag=inputs_imag_A)
        inputs_complex_B = tf.complex(real=inputs_real_B, imag=inputs_imag_B)
        # AWGN channel -> phase offsets
        if self.channel_type == 'AWGN':
            phase_offsetsA = tf.random.uniform(shape=(1,), minval=0, maxval=361, dtype=tf.int32)
            phase_offsetsA = tf.cast(phase_offsetsA, dtype=tf.float32)
            # phase_offsetsA = 0.
            phase_offsetsB = tf.random.uniform(shape=(1,), minval=0, maxval=361, dtype=tf.int32)
            phase_offsetsB = tf.cast(phase_offsetsB, dtype=tf.float32)
            # phase_offsetsB = 0.
            hA_complex = tf.exp(tf.complex(real=0., imag=math.pi * phase_offsetsA / 180))
            hB_complex = tf.exp(tf.complex(real=0., imag=math.pi * phase_offsetsB / 180))
        # Rayleigh channel
        elif self.channel_type == 'Rayleigh':
            pass
        # noise
        n_real = tf.random.normal(shape=tf.shape(inputs_complex_A), mean=0.0, stddev=std, dtype=tf.float32)
        n_imag = tf.random.normal(shape=tf.shape(inputs_complex_A), mean=0.0, stddev=std, dtype=tf.float32)
        noise = tf.complex(real=n_real, imag=n_imag)
        # received signal y
        hAx = tf.multiply(hA_complex, inputs_complex_A)
        hBx = tf.multiply(hB_complex, inputs_complex_B)
        hx = tf.add(hAx, hBx)
        y_complex = tf.add(hx, noise)
        # reshape
        y_real = tf.math.real(y_complex)
        y_imag = tf.math.imag(y_complex)
        y_real = tf.expand_dims(y_real, axis=-1)
        y_imag = tf.expand_dims(y_imag, axis=-1)
        output = tf.concat([y_real, y_imag], axis=-1)

        return output

    def get_config(self):
        base_config = super().get_config()
        base_config['snr_db'] = self.snr_db
        base_config['channel_type'] = self.channel_type
        base_config['phase_offsets'] = self.phase_offsets

        return base_config


# relay -> node cahnnel
class ChannelLayer(layers.Layer):
    def __init__(self, snr_db, channel_type: str='AWGN', name: str=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.snr_db = snr_db
        self.channel_type = channel_type

    def call(self, inputs, *args, **kwargs):
        # noise std
        Es = 1
        EsN0 = 10 ** (self.snr_db / 10)
        N0 = Es / EsN0
        sigma = math.sqrt(N0 / 2)
        std = tf.constant(value=sigma, dtype=tf.float32)
        # signal
        inputs_real = inputs[:, :, 0]
        inputs_imag = inputs[:, :, 1]
        inputs_complex = tf.complex(real=inputs_real, imag=inputs_imag)
        # AWGN channel
        if self.channel_type == 'AWGN':
            h_complex = tf.complex(real=1., imag=0.)
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
        # reshape
        y_real = tf.math.real(y_complex)
        y_imag = tf.math.imag(y_complex)
        y_real = tf.expand_dims(y_real, axis=-1)
        y_imag = tf.expand_dims(y_imag, axis=-1)
        output = tf.concat([y_real, y_imag], axis=-1)

        return output

    def get_config(self):
        base_config = super().get_config()
        base_config['snr_db'] = self.snr_db
        base_config['channel_type'] = self.channel_type

        return base_config


# ResBlk -> Tx
class ResidualBlockTx(layers.Layer):

    def __init__(self, out_channel, strides: int=1, downsample: bool=False, name: str=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.out_channel = out_channel
        self.strides = strides
        self.downsample = downsample
        if self.downsample:
            self.Conv2D_0 = layers.Conv2D(filters=self.out_channel, kernel_size=1, 
                                            strides=self.strides, padding='same', activation=None)
            self.BatchNorm_0 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.Conv2D_1 = layers.Conv2D(filters=self.out_channel, kernel_size=3, 
                                            strides=self.strides, padding='same', activation=None)
        self.Conv2D_2 = layers.Conv2D(filters=self.out_channel, kernel_size=3, 
                                            strides=1, padding='same',activation=None)
        self.BatchNorm_1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.BatchNorm_2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.ELU = layers.ELU()
        self.Add = layers.Add()

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        if self.downsample:
            residual = self.Conv2D_0(inputs)
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


# ResBlk -> R
class ResidualBlock(layers.Layer):

    def __init__(self, out_channel, strides: int=1, downsample: bool=False, name: str=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.out_channel = out_channel
        self.strides = strides
        self.downsample = downsample
        if self.downsample:
            self.Conv1D_0 = layers.Conv1D(filters=self.out_channel, kernel_size=1, 
                                            strides=self.strides, padding='same', activation=None)
            self.BatchNorm_0 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.Conv1D_1 = layers.Conv1D(filters=self.out_channel, kernel_size=3, 
                                        strides=self.strides, padding='same', activation=None)
        self.Conv1D_2 = layers.Conv1D(filters=self.out_channel, kernel_size=3, 
                                        strides=1, padding='same', activation=None)
        self.BatchNorm_1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.BatchNorm_2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.ELU = layers.ELU()
        self.Add = layers.Add()

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        if self.downsample:
            residual = self.Conv1D_0(inputs)
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


# ResBlk -> Rx
class ResidualBlockRx(layers.Layer):

    def __init__(self, out_channel, strides: int=1, upsample: bool=False, name: str=None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.out_channel = out_channel
        self.strides = strides
        self.upsample = upsample
        if self.upsample:
            self.Conv2DTrans_0 = layers.Conv2DTranspose(filters=self.out_channel, kernel_size=1, 
                                                        strides=self.strides, padding='same', activation=None)
            self.BatchNorm_0 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.Conv2DTrans_1 = layers.Conv2DTranspose(filters=self.out_channel, kernel_size=3, 
                                                        strides=self.strides, padding='same', activation=None)
        self.Conv2DTrans_2 = layers.Conv2DTranspose(filters=self.out_channel, kernel_size=3,
                                                        strides=1, padding='same', activation=None)
        self.BatchNorm_1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.BatchNorm_2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.ELU = layers.ELU()
        self.Add = layers.Add()

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        if self.upsample:
            residual = self.Conv2DTrans_0(inputs)
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
        base_config['upsample'] = self.upsample

        return base_config


# semantic encoder
class SemanticEncoder(layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(SemanticEncoder, self).__init__(name=name, **kwargs)
        self.Conv2D = layers.Conv2D(filters=4, kernel_size=3, strides=1, padding='same', activation='elu')
        self.ResBlk_1 = ResidualBlockTx(out_channel=8, strides=2, downsample=True)
        self.ResBlk_2 = ResidualBlockTx(out_channel=16, strides=2, downsample=True)

    def call(self, inputs, *args, **kwargs):
        x = self.Conv2D(inputs)
        x = self.ResBlk_1(x)
        x = self.ResBlk_2(x)

        return x


# channel encoder
class ChannelEncoder(layers.Layer):

    def __init__(self, num_symbol, name=None, **kwargs):
        super(ChannelEncoder, self).__init__(name=name, **kwargs)
        self.ResBlk_1 = ResidualBlockTx(out_channel=32, strides=1, downsample=True)
        self.ResBlk_2 = ResidualBlockTx(out_channel=32, strides=1, downsample=False)
        self.Flatten = layers.Flatten()
        self.Dense = layers.Dense(units=2*num_symbol, activation=None, use_bias=True)
        self.Reshape = layers.Reshape((-1, 2))

    def call(self, inputs, *args, **kwargs):
        x = self.ResBlk_1(inputs)
        x = self.ResBlk_2(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        x = self.Reshape(x)
        # power nrom
        x_norm = tf.math.sqrt(tf.cast(x.shape[1], tf.float32) / 2.0) * tf.math.l2_normalize(x, axis=1)

        return x_norm


# semantic phycisal-layer network coding
class SemanticPNC(layers.Layer):

    def __init__(self, num_symbol, name=None, **kwargs):
        super(SemanticPNC, self).__init__(name=name, **kwargs)
        self.Norm = layers.LayerNormalization(axis=1, epsilon=1e-5, center=True, scale=True)
        self.ResBlk_1 = ResidualBlock(out_channel=32, strides=1, downsample=True)
        self.ResBlk_2 = ResidualBlock(out_channel=32, strides=1, downsample=False)
        self.Flatten = layers.Flatten()
        self.Dense = layers.Dense(units=2*num_symbol, activation=None, use_bias=True)
        self.Reshape = layers.Reshape((-1,2))

    def call(self, inputs, *args, **kwargs):
        x = self.Norm(inputs)
        x = self.ResBlk_1(x)
        x = self.ResBlk_2(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        x = self.Reshape(x)
        # Power Norm
        x_norm = tf.math.sqrt(tf.cast(x.shape[1], tf.float32) / 2.0) * tf.math.l2_normalize(x, axis=1)

        return x_norm


# channel decoder
class ChannelDecoder(layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(ChannelDecoder, self).__init__(name=name, **kwargs)
        self.Concatenate = layers.Concatenate(axis=1)
        self.Flatten = layers.Flatten()
        self.Dense_1 = layers.Dense(units=7 * 7 * 16, activation=None, use_bias=True)
        self.Reshape = layers.Reshape((7, 7, 16))
        self.ResBlk_1 = ResidualBlockRx(out_channel=32, strides=1, upsample=True)
        self.ResBlk_2 = ResidualBlockRx(out_channel=16, strides=1, upsample=True)

    def call(self, inputs, *args, **kwargs):
        Concat = self.Concatenate(inputs)
        x = self.Flatten(Concat)
        x = self.Dense_1(x)
        # x = self.Dense_2(x)
        x = self.Reshape(x)
        x = self.ResBlk_1(x)
        x = self.ResBlk_2(x)

        return x


# semantic decoder
class SemanticDecoder(layers.Layer):

    def __init__(self, name=None, **kwargs):
        super(SemanticDecoder, self).__init__(name=name, **kwargs)

        self.ResBlk_1 = ResidualBlockRx(out_channel=8, strides=2, upsample=True)
        self.ResBlk_2 = ResidualBlockRx(out_channel=4, strides=2, upsample=True)
        self.Flatten = layers.Flatten()
        self.TransConv2D = layers.Conv2DTranspose(filters=1, kernel_size=3, 
                                                    strides=1, padding='same', activation=None)
        self.Sigmoid = layers.Activation('sigmoid')

    def call(self, inputs, *args, **kwargs):

        x = self.ResBlk_1(inputs)
        x = self.ResBlk_2(x)
        x = self.TransConv2D(x)
        x = self.Sigmoid(x)

        return x


# MaskedAutoencoder
class SemanticTWRC(Model):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)

        # --------------------------------------------------------------------------
        # transmitter
        self.SE_A = SemanticEncoder(name='SemanticEncoder_A')
        self.CE_A = ChannelEncoder(num_symbol=args.num_symbol_node, name='ChannelEncoder_A')
        self.SE_B = SemanticEncoder(name='SemanticEncoder_B')
        self.CE_B = ChannelEncoder(num_symbol=args.num_symbol_node, name='ChannelEncoder_B')
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # relay and channel
        self.channel_relay = ChannelLayerRelay(snr_db=args.snr_train_dB_up, channel_type=args.channel_type,name='relay_channel')
        self.SC_PNC = SemanticPNC(num_symbol=args.num_symbol_relay, name='Semantic_PNC')
        self.channel_A = ChannelLayer(snr_db=args.snr_train_dB_down, channel_type=args.channel_type, name='channel_A')
        self.channel_B = ChannelLayer(snr_db=args.snr_train_dB_down, channel_type=args.channel_type, name='channel_B')
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # receiver
        self.SD_A = SemanticDecoder(name='SemanticDecoder_A')
        self.CD_A = ChannelDecoder(name='ChannelDecoder_A')
        self.SD_B = SemanticDecoder(name='SemanticDecoder_B')
        self.CD_B = ChannelDecoder(name='ChannelDecoder_B')
        # --------------------------------------------------------------------------

    # Tx_A
    def transmitter_A(self, x):
        x = self.SE_A(x)
        x = self.CE_A(x)

        return x
    # Tx_B
    def transmitter_B(self, x):
        x = self.SE_B(x)
        x = self.CE_B(x)

        return x

    # R
    def relay(self, x):
        x = self.SC_PNC(x)

        return x

    # Rx_A
    def receiver_A(self, x):
        x = self.CD_A(x)
        x = self.SD_A(x)

        return x
    # Rx_B
    def receiver_B(self, x):
        x = self.CD_B(x)
        x = self.SD_B(x)

        return x

    def call(self, inputs, training=None):
        # [B, h, w, 3] -> [B, -1, 2]
        x_A = self.transmitter_A(inputs[0])
        x_B = self.transmitter_B(inputs[1])

        # [B, -1, 2] -> [B, -1, 2]
        y_R = self.channel_relay([x_A, x_B])
        x_R = self.relay(y_R)

        # [B, -1, 2] -> [B, h, w, 3]
        y_A = self.channel_A(x_R)
        rec_A = self.receiver_A([y_A, x_A])
        y_B = self.channel_B(x_R)
        rec_B = self.receiver_B([y_B, x_B])

        return rec_B, rec_A


# build model -> mnist
def semantic_twrc(args, **kwargs):
    model = SemanticTWRC(args)
    return model
