
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

def generate_ds(data_root: str,
                train_im_height: int = 224,
                train_im_width: int = 224,
                val_im_height: int = None,
                val_im_width: int = None,
                batch_size: int = 8,
                val_rate: float = 0.1,
                cache_data: bool = False):
    """
    读取划分数据集，并生成训练集和验证集的迭代器
    :param data_root: 数据根目录
    :param train_im_height: 训练输入网络图像的高度
    :param train_im_width:  训练输入网络图像的宽度
    :param val_im_height: 验证输入网络图像的高度
    :param val_im_width:  验证输入网络图像的宽度
    :param batch_size: 训练使用的batch size
    :param val_rate:  将数据按给定比例划分到验证集
    :param cache_data: 是否缓存数据
    :return:
    """
    assert train_im_height is not None
    assert train_im_width is not None
    if val_im_width is None:
        val_im_width = train_im_width
    if val_im_height is None:
        val_im_height = train_im_height

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Configure dataset for performance
    def configure_for_performance(ds,
                                  shuffle_size: int,
                                  shuffle: bool = False,
                                  cache: bool = False):
        if cache:
            ds = ds.cache()  # 读取数据后缓存至内存
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_size)  # 打乱数据顺序
        ds = ds.batch(batch_size)                      # 指定batch size
        ds = ds.prefetch(buffer_size=AUTOTUNE)         # 在训练的同时提前准备下一个step的数据
        return ds
    (x_train, _), (x_test, _) = keras.datasets.cifar100.load_data(label_mode='coarse')
    x_train = tf.image.rgb_to_grayscale(x_train.astype(np.float32)/255.)
    x_test = tf.image.rgb_to_grayscale(x_test.astype(np.float32)/255.)
    x_train_A = tf.random.shuffle(x_train)
    x_train_B = tf.random.shuffle(x_train)
    x_test_A = tf.random.shuffle(x_test)
    x_test_B = tf.random.shuffle(x_test)

    train_ds_A = tf.data.Dataset.from_tensor_slices((x_train_A, x_train_B))
    total_train = len(x_train)
    train_ds_A = configure_for_performance(train_ds_A, total_train, shuffle=True, cache=cache_data)

    train_ds_B = tf.data.Dataset.from_tensor_slices((x_train_B, x_train_B))
    total_train = len(x_train)
    train_ds_B = configure_for_performance(train_ds_B, total_train, shuffle=True, cache=cache_data)


    val_ds_A = tf.data.Dataset.from_tensor_slices((x_test_A, x_test_B))
    total_val = len(x_test)
    val_ds_A = configure_for_performance(val_ds_A, total_val, cache=False)

    val_ds_B = tf.data.Dataset.from_tensor_slices((x_test_B, x_test_B))
    total_val = len(x_test)
    val_ds_B = configure_for_performance(val_ds_B, total_val, cache=False)

    return train_ds_A, train_ds_B, val_ds_A, val_ds_B


# if __name__ == '__main__':
#     generate_ds(data_root='', 
#                 train_im_height=32,
#                 train_im_width=32,
#                 batch_size=8)