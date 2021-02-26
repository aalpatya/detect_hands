import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import time
import tensorflow as tf
import cv2
import numpy as np
import queue
import sounddevice as sd 

from utils import label_map_util
from utils import visualization_utils

########################################################################
# Audio properties
FS = 4410
# Base waveform of length FS to step through with base_wave_ptr
x = np.arange(FS)
base_wave = np.sin(2*np.pi*x/FS)
base_wave_ptr = 0

# Initialise freq and amp to 0
freq_in, amp_in = 0, 0.

# set freq and amp limits
MIN_FREQ, MAX_FREQ = 250, 1200
MIN_AMP, MAX_AMP = 0.0, 0.4
########################################################################
# Model information
MODEL_NAME = 'ssd_mobilenet_v2_fpn_320'
PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(), 'model_data', MODEL_NAME, 'saved_model')
PATH_TO_LABELS = os.path.join(os.getcwd(), 'model_data', MODEL_NAME, 'label_map.pbtxt')
print("Loading saved model ...")
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print("Model Loaded!")

# Load label map and obtain class names and ids
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
category_index = label_map_util.create_category_index(
    label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=1, use_display_name=True
    )
)

def visualise_on_image(image, bboxes, labels, scores, thresh):
    (h, w, d) = image.shape
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > thresh:
            xmin, ymin = int(bbox[1]*w), int(bbox[0]*h)
            xmax, ymax = int(bbox[3]*w), int(bbox[2]*h)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(image, f"{label}: {int(score*100)} %", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return image

# Create Queue for storing bounding box centroids
q = queue.Queue(maxsize=5)
cx, cy = 0, 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def audio_callback(outdata, frames, time, status):
    """ Gets freq and amp information from the queue and creates 
    samples to play from the base waveform """
    global base_wave_ptr, base_wave, freq_in, amp_in

    # Step through the base waveform in step size of desired freq
    for i in range(frames):
        outdata[i] = amp_in * base_wave[base_wave_ptr]
        base_wave_ptr = (base_wave_ptr + freq_in) % FS


with sd.OutputStream(channels=1, callback=audio_callback, samplerate=FS):
    tic = time.time()    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error reading frame from camera. Exiting ...')
            break
        
        frame = cv2.flip(frame, 1)
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        # The model expects a batch of images, so also add an axis with `tf.newaxis`.
        input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

        # Pass frame through detector
        detections = detect_fn(input_tensor)

        # Detection parameters
        score_thresh = 0.4
        max_detections = 4

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        scores = detections['detection_scores'][0, :max_detections].numpy()
        bboxes = detections['detection_boxes'][0, :max_detections].numpy()
        labels = detections['detection_classes'][0, :max_detections].numpy().astype(np.int64)
        labels = [category_index[n]['name'] for n in labels]

        max_score_idx = np.argmax(scores)
        if scores[max_score_idx] >= score_thresh:
            cx = (bboxes[max_score_idx,1]+bboxes[max_score_idx,3])/2
            cy = (bboxes[max_score_idx,0]+bboxes[max_score_idx,2])/2
        else:
            cy = cy * 0.6

        # Display detections
        visualise_on_image(frame, bboxes, labels, scores, score_thresh)

        toc = time.time()
        fps = int(1/(toc - tic))
        tic = toc
        cv2.putText(frame, f"FPS: {fps}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
        cv2.imshow("Hands Theremin", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        
        # Get mouse x,y position and convert to freq and amplitude
        freq_in = int(cx * (MAX_FREQ-MIN_FREQ) + MIN_FREQ)
        amp_in = cy * (MAX_AMP-MIN_AMP)


print("Exiting ...")
cap.release()
cv2.destroyAllWindows()