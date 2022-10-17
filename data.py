import tensorflow as tf
import tensorflow_datasets as tfds
from keras.applications import imagenet_utils
from keras_cv.layers import preprocessing


def get_augmentations(config):
    normalize = tf.keras.layers.Lambda(lambda x: imagenet_utils.preprocess_input(x, mode="tf"))
    flip_and_jitter = preprocessing.Augmenter(
        [
            preprocessing.RandomFlip(mode=preprocessing.random_flip.HORIZONTAL),
            preprocessing.MaybeApply(
                preprocessing.RandomColorJitter(
                    (0, 255), brightness_factor=0.4, contrast_factor=0.4, saturation_factor=0.2, hue_factor=0.1
                ),
                0.8,
            ),
            preprocessing.MaybeApply(preprocessing.Grayscale(output_channels=3), 0.2),
        ]
    )

    global_crop_1 = preprocessing.Augmenter(
        [
            preprocessing.RandomCropAndResize(
                (224, 224), crop_area_factor=config["global_crops_scale"], aspect_ratio_factor=((3 / 4, 4 / 3))
            ),
            flip_and_jitter,
            preprocessing.RandomGaussianBlur(kernel_size=1, factor=(0.5, 0.5)),  # not like paper
            normalize,
        ]
    )

    global_crop_2 = preprocessing.Augmenter(
        [
            preprocessing.RandomCropAndResize(
                (224, 224), crop_area_factor=config["global_crops_scale"], aspect_ratio_factor=((3 / 4, 4 / 3))
            ),
            flip_and_jitter,
            preprocessing.MaybeApply(
                preprocessing.RandomGaussianBlur(kernel_size=1, factor=(0.5, 0.5)), 0.1
            ),  # not like paper
            preprocessing.MaybeApply(preprocessing.Solarization((0, 255), threshold_factor=128), 0.2),
            normalize,
        ]
    )

    local_crop = preprocessing.Augmenter(
        [
            preprocessing.RandomCropAndResize(
                (96, 96), crop_area_factor=(0.05, 0.4), aspect_ratio_factor=((3 / 4, 4 / 3))
            ),
            flip_and_jitter,
            preprocessing.MaybeApply(preprocessing.RandomGaussianBlur(kernel_size=1, factor=(0.5, 0.5)), 0.5),
            normalize,
        ]
    )

    return global_crop_1, global_crop_2, local_crop


def dataset(config):
    global_crop_1, global_crop_2, local_crop = get_augmentations(config)

    ds = tfds.load("imagenette", as_supervised=True)["train"]
    ds = ds.map(lambda x, _: x)
    ds = ds.map(
        lambda image: {
            "global_crops": [global_crop_1(image), global_crop_2(image)],
            "local_crops": [local_crop(image) for _ in range(config["local_crops_number"])],
        },
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.batch(
        tf.distribute.get_strategy().num_replicas_in_sync * config["batch_size_per_gpu"], drop_remainder=True
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
