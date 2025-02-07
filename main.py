from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import torch
import base64
import json
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

app = FastAPI()

# Load YOLOv8 model (use 'yolov8s.pt' for better accuracy)
model = YOLO("yolov8n.pt")  

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # Receive base64-encoded image from client
            data = await websocket.receive_json()
            print("Data", data.get("image"))
            image_bytes = base64.b64decode(data.get("image"))
            np_img = np.frombuffer(image_bytes, dtype=np.uint8)
            print(np_img)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Invalid image"})
                continue

            # Run YOLOv8 inference
            results = model(frame)
            detections = json.loads(results[0].tojson())  # Convert JSON string to Python object
            
            # Extract necessary fields
            formatted_detections = [
                {"name": det["name"], "confidence": det["confidence"], "bbox": det["box"]}
                for det in detections
            ]

            await websocket.send_json({"detections": formatted_detections})

    except Exception as e:
        print(f"Client disconnected: {e}")
        await websocket.close()
    

@app.websocket("/ws-semantic")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # Receive base64-encoded image from client
            data = await websocket.receive_json()
            image_bytes = base64.b64decode(data.get("image"))
            np_img = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Invalid image"})
                continue

            # Convert OpenCV image (BGR) to PIL image (RGB) for BLIP
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Generate caption using BLIP
            inputs = blip_processor(images=pil_image, return_tensors="pt")
            caption_ids = blip_model.generate(**inputs)
            caption = blip_processor.batch_decode(caption_ids, skip_special_tokens=True)[0]

            # Run YOLOv8 inference
            results = model(frame)
            detections = json.loads(results[0].tojson())  # Convert JSON string to Python object

            # Extract necessary fields
            formatted_detections = [
                {"name": det["name"], "confidence": det["confidence"], "bbox": det["box"]}
                for det in detections
            ]

            # Send response with both YOLO detections and BLIP caption
            await websocket.send_json({
                "caption": caption,
                "detections": formatted_detections
            })

    except Exception as e:
        print(f"Client disconnected: {e}")
        await websocket.close()