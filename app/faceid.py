# App layer
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Build app and layout

class CamApp(App):
    
    def build(self):
        # Main layout components
        #self.title = Label(text = "Facial Recognition App", bold = True)
        self.web_cam = Image(size_hint = (1, .8))
        self.button = Button(text = "Verify", on_press = self.verify, size_hint = (1, .1))
        self.verification_label = Label(text = "Verification Uninitiated", size_hint = (1, .1))
        
        # Add items to layout
        layout = BoxLayout(orientation = "vertical")
        #layout.add_widget(self.title)
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        
        # load tensorflow model
        self.model = tf.keras.models.load_model("siamese_model")
        
        # setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout
    
    def update(self, *args):
        
        # read drame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        
        # flip horizontally and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size = (frame.shape[1], frame.shape[0]), colorfmt = "bgr")
        img_texture.blit_buffer(buf, colorfmt = "bgr", bufferfmt = "ubyte")
        self.web_cam.texture = img_texture
    
    def preprocess(self, file_path):
        """Load img from file_path and resize to 100*100 and scale image between 0 to 1
        
        Args:
            file_path (_type_): path to image
        
        Returns:
            img: return image after preprocessing
        """
        
        # read img from file path
        byte_img = tf.io.read_file(file_path)
        # load image using decode jpeg
        img = tf.io.decode_jpeg(byte_img)
        # Preprocessing steps - resize to 100*100*3 (similar to research paper)
        img = tf.image.resize(img, (100, 100))
        # scale img between 0 to 1
        img = img / 255.0
        
        return img
    
    # detection threshold: What the limit is before our prediction is considered positive
    # verifiacation threshold: What propostion of predictions needs to be positive for a match
    def verify(self,*args):
        # specify thresholds
        detection_threshold = 0.9
        verification_threshold = 0.9
        
        # capture input image from our webcam
        SAVE_PATH = os.path.join("application_data", "input_image", "input_image.jpg")
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)
        
        
        # build results array
        results_lst = []
        
        for image in os.listdir(os.path.join("application_data", "verification_images")):
            input_img = self.preprocess(os.path.join("application_data", "input_image", "input_image.jpg"))
            validation_img = self.preprocess(os.path.join("application_data", "verification_images", image))
            
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results_lst.append(result)
        
        # detection threshold
        detection = np.sum(np.array(results_lst) > detection_threshold)
        
        # verification threshold
        verification = detection / len(os.listdir(os.path.join("application_data", "verification_images")))
        verified = verification > verification_threshold
        
        # set verification label
        self.verification_label.text = "Verified" if verification == True else "Not Verified"
        
        return results_lst, verified
    
    

if __name__=='__main__':
    CamApp().run()