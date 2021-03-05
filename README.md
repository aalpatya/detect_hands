# SSD on CPU using subset of Egohands Dataset

This is a hands detector that uses the SSD_mobilenet_v2_fpn_320 network.
It has been trained with transfer learning, on a small subset (about 300 images) from the EgoHands dataset:
    http://vision.soic.indiana.edu/projects/egohands/

The dataset has been cleaned (sometimes bounding boxes in images had x_min = x_max, and y_min = y_max).
Tutorial: https://aalpatya.medium.com/train-an-object-detector-using-tensorflow-2-object-detection-api-in-2021-a4fed450d1b9

Acknowledgements:
- https://github.com/molyswu/hand_detection/blob/temp/hand_detection/egohands_dataset_clean.py (I modified this)
- https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py (I modified this)

Requires tensorflow 2, (tested on tensorflow 2.4.1)
and OpenCV 4 (tested on OpenCV 4.2.1)

# Live Hand Detection only
Run on python3 with:
    python3 webcam_detect_hands.py

# Live Hand Theremin
python3 theremin_hands.py

See it in action by clicking this thumbnail: 

[![Hand Theremin](http://img.youtube.com/vi/3Kw0j-96lWc/0.jpg)](http://www.youtube.com/watch?v=3Kw0j-96lWc "Hand Theremin")

(https://www.youtube.com/watch?v=3Kw0j-96lWc)

# Live Mouse Theremin
python3 theremin_mouse.py


TODO: Put camera frame reading and processing into multi-threading
