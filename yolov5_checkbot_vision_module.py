import os
import uuid
import shutil
import yaml
from bson.objectid import ObjectId
import sys
from shutil import copyfile

import train
import detect

sys.path.append("../../../")
from checkbot_vision_module import CheckbotVisionModule
from cbutils import cb_settings


class yolov5CheckbotVisionModule(CheckbotVisionModule):
    def __init__(self):
        super(CheckbotVisionModule, self).__init__()

    def data_loader(self):
        return super().data_loader()

    def custom_changes(self):
        """
        required dataset format

        YOLOV5_data_dir
            /obj_train_data/images
            /obj_train_data/labels
            /obj_valid_data/images
            /obj_valid_data/labels
        """

        data_dir = os.path.join(cb_settings.ROOT_PATH, cb_settings.DATASET_LOCATION,
                                "YOLOV5_" + str(uuid.uuid4()))
        os.mkdir(data_dir)
        print(data_dir)

        types = ['obj_train_data', 'obj_valid_data']

        for type in types:
            destination_images_dir = os.path.join(data_dir, type, "images")
            os.makedirs(destination_images_dir)

            destination_labels_dir = os.path.join(data_dir, type, "labels")
            os.makedirs(destination_labels_dir)

            label_files = os.listdir(os.path.join(self.labels_path, type))
            image_files = [x.replace(".txt", ".jpg") for x in label_files]

            for image, label in zip(image_files, label_files):
                copyfile(os.path.join(self.images_path, "dataset/images",
                                      image),
                         os.path.join(destination_images_dir, image))
                copyfile(os.path.join(self.labels_path, type, label),
                         os.path.join(destination_labels_dir, label))

        """
        update yaml file with new changes
        """
        filepath = "visionModules/objectDetection/yolov5/data/coco128.yaml"
        with open(filepath) as f:
            doc = yaml.load(f)
        doc['path'] = data_dir
        doc['nc'] = len(self.labels)
        doc['names'] = self.labels
        with open(filepath, 'w') as f:
            yaml.dump(doc, f)
        pass

    def update_model_store(self, vision_model_path, accuracy):
        return super().update_model_store(vision_model_path, accuracy)

    def update_training_status(self, status):
        return super().update_training_status(status)

    def train(self, model_id, inspection_code, checkbot_id):
        self.model_id = model_id
        self.inspection_code = inspection_code
        self.checkbot_id = checkbot_id

        self.labels, self.images_path, self.labels_path = self.data_loader()
        self.custom_changes()
        vision_model_path, accuracy = train.run(epochs=2)
        self.update_model_store(vision_model_path, accuracy)
        self.update_training_status("completed")
        self.update_trained_model_in_inspection_plan()
        print(vision_model_path)
        print(accuracy)
        print("training successfull...")
        print("_"*20)

    def infer(self, inspection_tranx_id, image_filename, vision_model_path):
        self.inspection_tranx_id = inspection_tranx_id
        prediction = detect.run(weights=vision_model_path,
                                source=image_filename)
        print(prediction)
        self.update_prediction(prediction)

    def update_prediction(self, prediction):
        return super().update_prediction(prediction)


def train_module(model_id, inspection_code, checkbot_id):
    mod = yolov5CheckbotVisionModule()
    mod.train(model_id, inspection_code, checkbot_id)
    return model_id


def infer_module(inspection_tranx_id, image_filename, vision_model_path):
    mod = yolov5CheckbotVisionModule()
    mod.infer(inspection_tranx_id, image_filename, vision_model_path)
    return inspection_tranx_id