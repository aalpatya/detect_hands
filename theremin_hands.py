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
FS = 4410
# base waveform of length FS to step through with base_wave_ptr
x = np.arange(FS)
base_wave = np.sin(2*np.pi*x/FS)
base_wave_ptr = 0

# Initialise freq and amp to 0 (if queue is empty, the last values are used)
f_prev = 0
amp_prev = 0

# set freq and amp limits
MIN_FREQ, MAX_FREQ = 300, 900
MIN_AMP, MAX_AMP = 0.0, 0.4
########################################################################
# SSD Model parameters
MODEL_NAME = 'ssd_mobilenet_v2_fpn_320'
PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(), 'model_data', MODEL_NAME, 'saved_model')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(os.getcwd(), 'model_data', MODEL_NAME, 'label_map.pbtxt')
NUM_CLASSES = 1

print("Loading saved model ...")
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print("Model Loaded!")

# Load label map and obtain class names and ids
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Create Queue for storing bounding box centroids
q = queue.Queue(maxsize=1)

def audio_callback(outdata, frames, time, status):
    """ Gets freq and amp information from the queue and creates 
    samples to play from the base waveform """
    global base_wave_ptr, base_wave, f_prev, amp_prev

    try:
        # Get values from the queue
        freq, amp = q.get_nowait()
        f_prev  = freq
        amp_prev = amp
    except queue.Empty:
        # If queue is empty, just play the last freq and amp values
        freq = f_prev
        amp = amp_prev
    
    # Step through the base waveform in step size of desired freq
    for i in range(frames):
        outdata[i] = amp * base_wave[base_wave_ptr]
        base_wave_ptr = (base_wave_ptr + freq) % FS


with sd.OutputStream(channels=1, callback=audio_callback, samplerate=FS):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    tic = time.time()    
    while True:
        ret, frame = cap.read()
        image_np = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        max_detections = 4
        detections = {key: value[0, :max_detections].numpy()
                    for key, value in detections.items()}
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        boxes = detections['detection_boxes']
        cx = (boxes[0,1]+boxes[0,3])/2
        cy = (boxes[0,0]+boxes[0,2])/2

        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=max_detections,
            min_score_thresh=.40)

        toc = time.time()
        fps = int(1/(toc - tic))
        tic = toc
        cv2.putText(image_np_with_detections, f"FPS: {fps}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
        cv2.imshow("img", image_np_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        
        # Get mouse x,y position and convert to freq and amplitude
        freq_in = int(cx * (MAX_FREQ-MIN_FREQ) + MIN_FREQ)
        amp_in = cy * (MAX_AMP-MIN_AMP)

        # Put freq and amplitude in the queue
        q.put_nowait([freq_in, amp_in])

print("Exiting ...")
cap.release()
cv2.destroyAllWindows()