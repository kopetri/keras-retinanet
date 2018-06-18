from .generator import Generator
import glm
import numpy as np
import os
from PyGLEngine.GLEngine import ImageRenderer


class GLEngineGenerator(Generator):
    def __init__(
            self,
            number_of_images,
            model_dir,
            skybox_dir,
            save_dir,
            img_width,
            img_height,
            **kwargs
    ):
        self.width = img_width
        self.height = img_height

        self.renderer = ImageRenderer(
            modelsDir=model_dir,
            skyboxDir=skybox_dir,
            saveDir=save_dir,
            enable_skybox=not skybox_dir == "",
            random=True,
            calculate_bounding_box=True,
            camera_distance=(1.5, 4.0),
            width=img_width,
            height=img_height,
            number_of_images=number_of_images
        )

        self.labels = self.renderer.class_names
        self.classes = {}
        for i, class_name in enumerate(self.labels):
            self.classes[class_name] = i
        with open(os.path.join(save_dir, "annotations.csv"), "w") as file:
            self.renderer.writeAllFrames(file)

        if len(self.renderer.frames) < self.renderer.number_of_images:
            print("len(self.renderer.frames) < self.renderer.number_of_images")
            print(str(self.renderer.number_of_images))
            print(str(len(self.renderer.frames)))

        super(GLEngineGenerator, self).__init__(**kwargs)

    def size(self):
        return self.renderer.number_of_images

    def num_classes(self):
        return self.renderer.nb_classes

    def image_aspect_ratio(self, image_index):
        return float(self.width) / float(self.height)

    def load_image(self, image_index):
        return self.renderer.frames[image_index]

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def load_annotations(self, image_index):
        annotation = self.renderer.annotations[image_index]
        label = self.renderer.labels[image_index]
        label = np.argmax(label)
        boxes = np.zeros((1, 5))
        boxes[0, 0] = float(annotation[0])
        boxes[0, 1] = float(annotation[1])
        boxes[0, 2] = float(annotation[2])
        boxes[0, 3] = float(annotation[3])
        boxes[0, 4] = label
        return boxes
