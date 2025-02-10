import numpy as np
import base64
import cv2
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from app.models.Image import Image

class ImageService:
    def __init__(self):
        self.yolo_model = YOLO("../yolov8n.pt")
        self.blip_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        self.blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

    def convert_data_to_bytes_base64(self, image_data):
        return base64.b64decode(image_data)
    
    def convert_to_buffer(self, image_bytes):
        return np.frombuffer(image_bytes, dtype=np.uint8)
    
    def convert_to_cv2_decode_frame(self, image_buffer_arr):
        return cv2.imdecode(image_buffer_arr, cv2.IMREAD_COLOR)
    
    def convert_bgr_to_rgb_image(self, frame):
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    def object_detection_yolo(self, frame):
        return self.yolo_model(frame)

