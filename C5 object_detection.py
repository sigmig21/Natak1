# -*- coding: utf-8 -*-
"""5 object detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1P7pF0rx95Vej24XnZz1zB6gxV42riZuA
"""

!pip install ultralytics
from ultralytics import YOLO
import cv2
model=YOLO('yolov8n.pt')

from google.colab import drive
drive.mount('/content/drive')

model.train(data='/content/drive/MyDrive/CVExamDatasets /object detection/Persian_Car_Plates_YOLOV8/data.yaml',
            epochs=30,
            imgsz=640,
            batch=16,
            workers=4,
            name='custom_yolo_model')

model = YOLO('/content/runs/detect/custom_yolo_model2/weights/best`.pt')
# Run predictions
results = model.predict(source='/content/drive/MyDrive/CVExamDatasets /object detection/Persian_Car_Plates_YOLOV8/test/images/209_png.rf.ce3ce3b5716fd200fb31591ffe9b700a.jpg', show=False)
import matplotlib.pyplot as plt
result_img = results[0].plot()
result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(result_img_rgb)
plt.axis('off')
plt.show()``