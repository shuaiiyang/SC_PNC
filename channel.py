# -*- coding: utf-8 -*-
# @Author  : Shuai_Yang
# @Time    : 2022/2/

import math
import tensorflow as tf

# node to relay channel
def channel_relay(inputs, snra_db, snrb_db, channel_type: str='AWGN', phase_offsets: int=0, **kwargs):
    # noise std
    Es = 1
    EsN0 = 10**(snra_db/10)
    N0 = Es/EsN0
    sigma = math.sqrt(N0/2)
    std = tf.constant(value=sigma, dtype=tf.float32)
    # signal
    inputs_real_A = inputs[0][:, :,0]
    inputs_imag_A = inputs[0][:, :,1]
    inputs_real_B = inputs[1][:, :,0]
    inputs_imag_B = inputs[1][:, :,1]
    inputs_complex_A = tf.complex(real=inputs_real_A, imag=inputs_imag_A)
    inputs_complex_B = tf.complex(real=inputs_real_B, imag=inputs_imag_B)
    # AWGN channel
    if channel_type == 'AWGN':
        offsets_A = tf.constant(value=0, dtype=tf.float32, shape=tf.shape(inputs_complex_A))
        hA_complex = tf.exp(tf.complex(real=tf.zeros(shape=tf.shape(inputs_complex_A), dtype=tf.float32), imag=math.pi*offsets_A/180))
        print(hA_complex[0,0])
        offsets_B = tf.constant(value=phase_offsets, dtype=tf.float32, shape=tf.shape(inputs_complex_B))
        hB_complex = tf.exp(tf.complex(real=tf.zeros(shape=tf.shape(inputs_complex_B), dtype=tf.float32), imag=math.pi*offsets_B/180))
        print(hB_complex[0,0])
    # Rayleigh channel
    elif channel_type == 'Rayleigh':
        pass
    # noise
    n_real = tf.random.normal(shape=tf.shape(inputs_complex_A), mean=0.0, stddev=std, dtype=tf.float32)
    n_imag = tf.random.normal(shape=tf.shape(inputs_complex_A), mean=0.0, stddev=std, dtype=tf.float32)
    noise = tf.complex(real=n_real, imag=n_imag)
    # power ratio
    p_a = tf.constant(value=1, dtype=tf.complex64)
    p_b = tf.constant(value=math.sqrt(10**((snrb_db-snra_db)/10)), dtype=tf.complex64)
    hAx = tf.multiply(hA_complex, inputs_complex_A)
    hBx = tf.multiply(hB_complex, inputs_complex_B)
    # received signal y
    hx = tf.add(p_a*hAx, p_b*hBx)
    y_complex = tf.add(hx, noise)
    # reshape
    y_real = tf.math.real(y_complex)
    y_imag = tf.math.imag(y_complex)
    y_real = tf.expand_dims(y_real, axis=-1)
    y_imag = tf.expand_dims(y_imag, axis=-1)
    output = tf.concat([y_real, y_imag], axis=-1)

    return output


# relay to node channel
def channel(inputs, snr_db, channel_type: str='AWGN', *args, **kwargs):
    Es = 1
    EsN0 = 10 ** (snr_db / 10)
    N0 = Es / EsN0
    sigma = math.sqrt(N0 / 2)
    std = tf.constant(value=sigma, dtype=tf.float32)
    inputs_real = inputs[:, :,0]
    inputs_imag = inputs[:, :,1]
    inputs_complex = tf.complex(real=inputs_real, imag=inputs_imag)
    # AWGN channel
    if channel_type == 'AWGN':
        h_complex = tf.complex(real=1., imag=0.)
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
    # reshape
    y_real = tf.math.real(y_complex)
    y_imag = tf.math.imag(y_complex)
    y_real = tf.expand_dims(y_real, axis=-1)
    y_imag = tf.expand_dims(y_imag, axis=-1)
    output = tf.concat([y_real, y_imag], axis=-1)

    return output

#
# if __name__ == '__main__':

