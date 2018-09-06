import sys
import h5py
from tqdm import tqdm, trange
import numpy as np
from six import raise_from
import csv
import cv2

from .generator import Generator
from .csv_generator import _read_classes, _open_for_csv


class H5PyGenerator(Generator):
    def size(self):
        return self.dataset_size

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.class_names[label]

    def image_aspect_ratio(self, image_index):
        return float(self.hdf5_dataset['image_shapes'][image_index][1]) / float(self.hdf5_dataset['image_shapes'][image_index][0])

    def load_image(self, image_index):
        image = self.hdf5_dataset['images'][image_index].reshape(self.hdf5_dataset['image_shapes'][image_index])
        image = image[..., ::-1]
        return image

    def load_annotations(self, image_index):
        annots = self.labels[image_index]
        boxes = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):
            boxes[idx, 0] = float(annot[1])
            boxes[idx, 1] = float(annot[2])
            boxes[idx, 2] = float(annot[3])
            boxes[idx, 3] = float(annot[4])
            boxes[idx, 4] = annot[0]

        return boxes

    def __init__(
            self,
            hdf5_dataset_path,
            csv_class_file,
            load_images_into_memory=False,
            verbose=True,
            **kwargs
    ):
        '''
                Loads an HDF5 dataset that is in the format that the `create_hdf5_dataset()` method
                produces.

                Arguments:
                    verbose (bool, optional): If `True`, prints out the progress while loading
                        the dataset.

                Returns:
                    None.
                '''
        # parse the provided class file
        try:
            with _open_for_csv(csv_class_file) as file:
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.class_names = {}
        for key, value in self.classes.items():
            self.class_names[value] = key
        self.hdf5_dataset = h5py.File(hdf5_dataset_path, 'r')
        self.dataset_size = len(self.hdf5_dataset['images'])
        self.dataset_indices = np.arange(self.dataset_size,
                                         dtype=np.int32)  # Instead of shuffling the HDF5 dataset or images in memory, we will shuffle this index list.

        if load_images_into_memory:
            self.images = []
            if verbose:
                tr = trange(self.dataset_size, desc='Loading images into memory', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.images.append(self.hdf5_dataset['images'][i].reshape(self.hdf5_dataset['image_shapes'][i]))

        if self.hdf5_dataset.attrs['has_labels']:
            self.labels = []
            labels = self.hdf5_dataset['labels']
            label_shapes = self.hdf5_dataset['label_shapes']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading labels', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.labels.append(labels[i].reshape(label_shapes[i]))

        if self.hdf5_dataset.attrs['has_image_ids']:
            self.image_ids = []
            image_ids = self.hdf5_dataset['image_ids']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading image IDs', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.image_ids.append(image_ids[i])

        if self.hdf5_dataset.attrs['has_eval_neutral']:
            self.eval_neutral = []
            eval_neutral = self.hdf5_dataset['eval_neutral']
            if verbose:
                tr = trange(self.dataset_size, desc='Loading evaluation-neutrality annotations', file=sys.stdout)
            else:
                tr = range(self.dataset_size)
            for i in tr:
                self.eval_neutral.append(eval_neutral[i])

        for key in list(self.hdf5_dataset.keys()):
            print(self.hdf5_dataset[key])
        super(H5PyGenerator, self).__init__(**kwargs)