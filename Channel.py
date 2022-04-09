# -*- coding: utf-8 -*-
# @Author  : Shuai_Yang
# @Time    : 2022/2/

import math
import tensorflow as tf


# Relay 信道层
def channel_relay(inputs, snra_db, snrb_db, channel_type, modulation='BPSK', phase_offsets=0, **kwargs):
    Eb = 1/2
    # SNR(dB) = Eb/N0(dB) + 10log(k)
    EbN0 = 10**(snra_db/10)
    N0 = Eb/EbN0
    sigma = math.sqrt(N0/2)
    std = tf.constant(value=sigma, dtype=tf.float32)
    inputs_real_A = inputs[0][:, 0:128]
    inputs_imag_A = inputs[0][:, 128:256]
    inputs_real_B = inputs[1][:, 0:128]
    inputs_imag_B = inputs[1][:, 128:256]
    inputs_complex_A = tf.complex(real=inputs_real_A, imag=inputs_imag_A)
    inputs_complex_B = tf.complex(real=inputs_real_B, imag=inputs_imag_B)
    # AWGN channel
    if channel_type == 'AWGN':
        # hA_complex = tf.ones(shape=tf.shape(inputs_real_A), dtype=tf.complex64)*tf.exp(tf.complex(real=0., imag=math.pi*0/180))
        hA_complex = tf.exp(tf.complex(real=0., imag=math.pi*0/180))
        # hB_complex = tf.ones(shape=tf.shape(inputs_real_A), dtype=tf.complex64)*tf.exp(tf.complex(real=0., imag=math.pi*phase_offsets/180))
        hB_complex = tf.exp(tf.complex(real=0., imag=math.pi*phase_offsets/180))
    # Rayleigh channel
    elif channel_type == 'Rayleigh':
        pass
    # noise
    n_real = tf.random.normal(shape=tf.shape(inputs_complex_A), mean=0.0, stddev=std, dtype=tf.float32)
    n_imag = tf.random.normal(shape=tf.shape(inputs_complex_A), mean=0.0, stddev=std, dtype=tf.float32)
    noise = tf.complex(real=n_real, imag=n_imag)
    # received signal y
    # power
    p_a = tf.constant(value=1, dtype=tf.complex64)
    p_b = tf.constant(value=math.sqrt(10**((snrb_db-snra_db)/10)), dtype=tf.complex64)
    hAx = tf.multiply(hA_complex, inputs_complex_A)
    hBx = tf.multiply(hB_complex, inputs_complex_B)
    hx = tf.add(p_a*hAx, p_b*hBx)
    y_complex = tf.add(hx, noise)
    # y_complex = y_complex/p_b
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


# 信道
def channel(inputs, snr_db, channel_type, modulation='BPSK', *args, **kwargs):
    Eb = 1/2
    # SNR(dB) = Eb/N0(dB) + 10log(k)
    EbN0 = 10 ** (snr_db / 10)
    N0 = Eb / EbN0
    sigma = math.sqrt(N0 / 2)
    std = tf.constant(value=sigma, dtype=tf.float32)
    inputs_real = inputs[:, 0:128]
    inputs_imag = inputs[:, 128:256]
    inputs_complex = tf.complex(real=inputs_real, imag=inputs_imag)
    # AWGN channel
    if channel_type == 'AWGN':
        h_complex = tf.exp(tf.complex(real=0., imag=math.pi * 0 / 180))
    # Rayleigh channel
    elif channel_type == 'Rayleigh':
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

#
# if __name__ == '__main__':
#     Eb = 1 / 2
#     # SNR(dB) = Eb/N0(dB) + 10log(k)
#     EbN0 = 10 ** (0 / 10)
#     N0 = Eb / EbN0
#     sigma = math.sqrt(N0 / 2)
#     sigma = np.sqrt(N0 / 2)
#     sigma = np.sqrt(N0 / 2)
