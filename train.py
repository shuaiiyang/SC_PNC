# -*- coding: utf-8 -*-
# @Author  : Shuai_Yang
# @Time    : 2022/3/23
"""
Semantic PNC for train
"""

import glob
import sys
import math
import argparse
from tqdm import tqdm
import tensorflow as tf
from keras import optimizers, datasets

from utils import generate_ds, PeakSignalToNoiseRatio
from model import semantic_twrc as creat_model

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'


# define global parameters
def parse_args():
    parser = argparse.ArgumentParser(description="semantic communication empowered physical-layer network coding")
    
    # parameter of datasets
    parser.add_argument("--image_height", type=int, default=28, help="height of training and validation images")
    parser.add_argument("--image_width", type=int, default=28, help="width of training and validation images")
    parser.add_argument("--image_channel", type=int, default=1, help="channel of training and validation images")
    parser.add_argument("--val_rate", type=float, default=0.5, help="sample rate for validation")
    # parameter of model
    parser.add_argument("--num_symbol_node", type=int, default=256, help="the number of symbols sent by the node.")
    parser.add_argument("--num_symbol_relay", type=int, default=256, help="the number of symbols sent by relay.")
    # parameter of wireless channel
    parser.add_argument("--channel_type", type=str, default='AWGN', help="channel type during training。")
    parser.add_argument("--snr_train_dB_up", type=int, default=7, help="snr of node to relay in dB for training.")
    parser.add_argument("--snr_train_dB_down", type=int, default=7, help="snr of relay to node in dB for training.")
    # parameter of training
    parser.add_argument("--num_epochs", type=int, default=300, help="training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate.")
    parser.add_argument("--lr_decay", type=bool, default=False, help="whether to use decreasing learning rate.")
    args = parser.parse_args()
    
    return args


def main(args):

    # logs
    import datetime
    log_dir = "SC_PNC/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    import os
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    # data generator with data augmentation
    train_ds, val_ds = generate_ds(data_root=datasets.mnist, 
                                    train_im_height=args.image_height, 
                                    train_im_width=args.image_width, 
                                    batch_size=args.batch_size, 
                                    val_rate=args.val_rate, 
                                    cache_data=True)

    # create model of SC-PNC
    model = creat_model(args)
    model.build(input_shape=[(1,args.image_height,args.image_width,args.image_channel),
                            (1,args.image_height,args.image_width,args.image_channel)])
    model.summary()
    if not os.path.exists("SC_PNC/save_weights"):
        # Create a folder to save weights
        os.makedirs("SC_PNC/save_weights")
    else:
        # load weight of SC PNC
        weights_path = 'SC_PNC/save_weights/model.ckpt'
        assert len(glob.glob(weights_path+'*')), "cannot find {}".format(weights_path)
        # model.load_weights(weights_path)

    # custom learning rate curve
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / args.num_epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
        new_lr = rate * args.lr

        # writing lr into tensorboard
        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)

        return new_lr

    # using keras low level api for training
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = optimizers.Adam(learning_rate=args.lr)

    train_loss_AB = tf.keras.metrics.Mean(name='train_loss_AB')
    train_loss_BA = tf.keras.metrics.Mean(name='train_loss_BA')
    train_psnr_AB = PeakSignalToNoiseRatio(name='train_psnr_AB')
    train_psnr_BA = PeakSignalToNoiseRatio(name='train_psnr_BA')

    val_loss_AB = tf.keras.metrics.Mean(name='val_loss_AB')
    val_loss_BA = tf.keras.metrics.Mean(name='val_loss_BA')
    val_psnr_AB = PeakSignalToNoiseRatio(name='val_psnr_AB')
    val_psnr_BA = PeakSignalToNoiseRatio(name='val_psnr_BA')


    @tf.function
    def train_step(train_images_A, train_images_B):
        with tf.GradientTape() as tape:
            # MSE -> user A,B
            output_img_B, output_img_A = model([train_images_A, train_images_B], training=True)
            # losses
            loss_AB = loss_object(train_images_A, output_img_B)
            loss_BA = loss_object(train_images_B, output_img_A)
            loss = loss_AB + loss_BA

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # update loss and acc
        train_loss_AB(loss_AB)
        train_loss_BA(loss_BA)
        train_psnr_AB(train_images_A, output_img_B)
        train_psnr_BA(train_images_B, output_img_A)


    @tf.function
    def val_step(val_images_A, val_images_B):
        output_img_B, output_img_A = model([val_images_A, val_images_B], training=False)

        loss_AB = loss_object(val_images_A, output_img_B)
        loss_BA = loss_object(val_images_B, output_img_A)

        val_loss_AB(loss_AB)
        val_loss_BA(loss_BA) 
        val_psnr_AB(val_images_A, output_img_B)
        val_psnr_BA(val_images_B, output_img_A)


    best_val_loss = 1.
    for epoch in range(args.num_epochs):
        # clear train history info
        train_loss_AB.reset_states()
        train_loss_BA.reset_states()
        train_psnr_AB.reset_states()
        train_psnr_BA.reset_states()
        # clear val history info
        val_loss_AB.reset_states()
        val_loss_BA.reset_states()
        val_psnr_AB.reset_states()
        val_psnr_BA.reset_states()
        # train
        train_bar = tqdm(train_ds, file=sys.stdout)
        for images_A, images_B in train_bar:
            train_step(images_A, images_B)
            # print train process
            train_bar.desc = "train epoch[{}/{}] loss_AB:{:.4f}, loss_BA:{:.4f}, psnr_AB:{:.4f}, psnr_BA:{:.4f}".format(
                            epoch + 1, 
                            args.num_epochs, 
                            train_loss_AB.result(), 
                            train_loss_BA.result(), 
                            train_psnr_AB.result(),
                            train_psnr_BA.result(),
                            )
        # update learning rate
        if args.lr_decay:
            optimizer.learning_rate = scheduler(epoch)
        # validate
        val_bar = tqdm(val_ds, file=sys.stdout)
        for images_A, images_B in val_bar:
            val_step(images_A, images_B)
            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss_AB:{:.4f}, loss_BA:{:.4f}, psnr_AB:{:.4f}, psnr_BA:{:.4f}".format(
                            epoch + 1, 
                            args.num_epochs, 
                            val_loss_AB.result(), 
                            val_loss_BA.result(), 
                            val_psnr_AB.result(),
                            val_psnr_BA.result(),
                            )

        # writing training loss and acc
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss_AB.result()+train_loss_BA.result(), epoch)

        # writing validation loss and acc
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss_AB.result()+val_loss_BA.result(), epoch)

        # only save best weights
        if val_loss_AB.result()+val_loss_BA.result()  < best_val_loss:
            best_val_loss = val_loss_AB.result()+val_loss_BA.result()
            save_name = "SC_PNC/save_weights/model.ckpt"
            model.save_weights(save_name, save_format="tf")
            print('save model!')


if __name__ == '__main__':
    args = parse_args()
    print("Called with args:", args)
    main(args)