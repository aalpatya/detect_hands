import cv2
import numpy as np 
import os

cap = cv2.VideoCapture(0)
PB_PATH = os.path.join('frozen_models', 'frozen_graph.pb')
# PBTXT_PATH = 'model_data\ssd_mobilenet_v2_fpn_320\label_map.pbtxt'
PBTXT_PATH = 'frozen_out.pbtxt'
CFG_PATH = os.path.join('model_data', 'pipeline.config')
tensorflowNet = cv2.dnn.readNetFromTensorflow(PB_PATH, PBTXT_PATH)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (300,300))
    tensorflowNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
    # Runs a forward pass to compute the net output
    networkOutput = tensorflowNet.forward()
    
    # Loop on the outputs
    for detection in networkOutput[0,0]:
        
        score = float(detection[2])
        if score > 0.2:
            
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
    
            #draw a red rectangle around detected objects
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
    
    # Show the image with a rectagle surrounding the detected objects 
    cv2.imshow('Image', frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()