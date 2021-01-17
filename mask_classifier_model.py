import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from data_loader import DataLoader
from datetime import date
import cv2
import numpy as np

class MaskClassifier():
    
    def __init__(self, data_loader, model="default"):
        assert isinstance(data_loader, DataLoader), "Invalid data loader."
        self.data_loader = data_loader
        if model == "default":
            self.model = Sequential([
                layers.experimental.preprocessing.Rescaling(1./255, input_shape=(128, 128, 3)),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(self.data_loader.num_classes)])
            
            self.model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                metrics=['accuracy'])
        else:
            self.model = tf.keras.models.load_model(model)
        self.model.summary()
        
    def train(self, epochs):
        history = self.model.fit(self.data_loader.training_data, validation_data=self.data_loader.validation_data, epochs=epochs)
        return history
    
    def predict(self, img):
        img = cv2.resize(img,(128,128))
        img = np.reshape(img,[1,128,128,3])
        predictions = self.model.predict(img)
        score = tf.nn.softmax(predictions[0])
        return score
    
    def predict_class(self, img):
        score = self.predict(img)
        return self.data_loader.class_names[np.argmax(score)]
    
    def classify(self, img):
        score = self.predict(img)
        print("The person is {} with a {:.2f} percent confidence.".format(self.data_loader.class_names[np.argmax(score)], 100 * np.max(score)))
    
    def save(self):
        today = date.today().strftime("%m-%d-%y")
        self.model.save("models/model_" + today)
        
        
        