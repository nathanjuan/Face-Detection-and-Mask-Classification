import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class DataLoader():
    def __init__(self, dataset_location, img_res = (128, 128)):
        self.dataset_path = pathlib.Path(dataset_location)
        self.img_res = img_res
        self.training_data = None
        self.validation_data = None
        self.class_names = None
        self.num_classes = 0
    
    def load_data(self, batch_size=30, validation_split=0.1):
        self.training_data = tf.keras.preprocessing.image_dataset_from_directory(directory=self.dataset_path, labels='inferred', validation_split=validation_split, subset="training", shuffle = True, seed=123, image_size=self.img_res, batch_size=batch_size)
        self.validation_data = tf.keras.preprocessing.image_dataset_from_directory(directory=self.dataset_path, labels='inferred', validation_split=validation_split, subset="validation", shuffle = True, seed=123, image_size=self.img_res, batch_size=batch_size)
        self.class_names = self.training_data.class_names
        self.num_classes = len(self.class_names)
        return self.training_data, self.validation_data
    
    def get_random_training_image(self):
        return self.get_image()
        
    def get_random_validation_image(self):
        return self.get_image("validation")
    
    def get_image(self, data_type = "training"):
        if data_type == "training":
            shuffled = self.training_data.shuffle(len(self.training_data))
        else:
            shuffled = self.validation_data.shuffle(len(self.validation_data))
        for images, labels in shuffled.take(1):
            i = np.random.randint(len(images))
            return images[i].numpy().astype("uint8"), self.class_names[labels[i]]
        
    def get_class_image(self, img_class):
        if img_class not in self.class_names:
            print("Class is not in dataset.")
            return None
        while True:
            img, label = self.get_image()
            if label == img_class:
                return img, label
    
    def show_image(self, img, label=""):
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
    
    def show_random_training(self):
        image, label = self.get_random_training_image()
        self.show_image(image, label)
        
    def show_random_validation(self):
        image, label = self.get_random_validation_image()
        self.show_image(image, label)
        
    def show_class_image(self, img_class):
        if img_class in self.class_names:
            image, label = self.get_class_image(img_class)
            self.show_image(image, label)
        else:
            print("Inputted class is not in the dataset. Choose from the following classes:" + str(self.class_names))
            
    def read_image(img_file):
        return mpimg.imread(img_file)
    
        