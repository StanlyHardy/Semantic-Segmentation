import math

from glob import glob

from keras.utils import Sequence

import numpy as np
import scipy.misc
from keras.preprocessing.image import load_img
from imgaug import augmenters as iaa


class DataSequence(Sequence):

    def __init__(self, data_dir, batch_size, image_shape):
        """
        Keras Sequence object to train a model on larger-than-memory data.
            @:param: data_dir: directory in which we have got the kitti images and the corresponding masks
            @:param: batch_size: define the number of training samples to be propagated.
            @:param: image_shape: shape of the input image
        """

        self.batch_size = batch_size
        self.image_shape = image_shape

        # define the image and mask paths
        self.image_paths = glob(data_dir + "/images/*.png")
        self.mask_paths = glob(data_dir + "data_dir/masks/*.png")

        # sort the image and mask paths to maintain the coherence
        self.image_paths = sorted(self.image_paths)
        self.mask_paths = sorted(self.mask_paths)

        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # alter it as per the labels

        self.color_dict = {0: (255, 0, 0),  # building
                           1: (0, 0, 255),  # road
                           2: (255, 255, 255)}  # background

        # create the augmentation pipeline
        self.aug_pipe = iaa.Sequential(
            [
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images

                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               ]),
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

    def __len__(self):
        """
        Number of batch in the Sequence.
        :return: The number of batches in the Sequence.
        """
        return int(math.ceil(len(self.image_paths) / float(self.batch_size)))

    def get_batch_images(self, idx, path_list):
        """

        :param idx: position of the image in the Sequence.
        :param path_list: list that consists of all the image paths
        :return: Retrieve the images in batches
        """
        # Fetch a batch of images from a list of paths
        for im in path_list[idx * self.batch_size: (1 + idx) * self.batch_size]:
            # load the image and resize
            image = load_img(im)
            image = scipy.misc.imresize(image, (self.image_shape[1], self.image_shape[0])) / 255.
            # augment the image
            image = self.aug_pipe.augment_image(image) / 255.
            return np.array([image])

    def get_batch_labels(self, idx, path_list):

        """
        Retrieve the masks in batches
        :param idx: position of the mask in the Sequence.
        :param path_list: list that consists of all the mask paths
        :return: mask labels
        """
        # iterate and map the mask labels for the respective images
        for im in path_list[idx * self.batch_size: (1 + idx) * self.batch_size]:
            # load the image and resize
            gt_image = load_img(im)
            gt_image = scipy.misc.imresize(gt_image, (self.image_shape[1], self.image_shape[0]))
            # augment the image
            gt_image = scipy.misc.imresize(gt_image, (self.image_shape[1], self.image_shape[0]))
            background_color = np.array([255, 0, 0])
            gt_bg = np.all(gt_image == background_color, axis=2)
            gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
            gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
            print(np.array([gt_image]).shape)

            return np.array([gt_image])

    def rgb_to_onehot(self, rgb_arr, color_dict):
        num_classes = len(color_dict)
        shape = rgb_arr.shape[:2] + (num_classes,)
        # print(shape)
        arr = np.zeros(shape, dtype=np.int8)
        for i, cls in enumerate(color_dict):
            arr[:, :, i] = np.all(rgb_arr.reshape((-1, 3)) == color_dict[i], axis=1).reshape(shape[:2])
        return arr

    def get_multi_batch_labels(self, idx, path_list):

        """
        Retrieve the masks in batches
        :param idx: position of the mask in the Sequence.
        :param path_list: list that consists of all the mask paths
        :return: mask labels
        """
        # iterate and map the mask labels for the respective images
        for im in path_list[idx * self.batch_size: (1 + idx) * self.batch_size]:
            # load the image and resize
            gt_image = load_img(im)
            gt_image = scipy.misc.imresize(gt_image, (self.image_shape[1], self.image_shape[0]))
            # # augment the image
            gt_image = self.rgb_to_onehot(gt_image, self.color_dict)

            return np.array([gt_image])

    def __getitem__(self, idx):
        """
        Retrieve the mask and the image in batches at position idx
        :param idx: position of the batch in the Sequence.
        :return: batches of image and the corresponding mask
        """
        batch_x = self.get_batch_images(idx, self.image_paths)
        batch_y = self.get_multi_batch_labels(idx, self.mask_paths)
        return batch_x, batch_y
