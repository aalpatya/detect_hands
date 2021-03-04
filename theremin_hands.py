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
MIN_FREQ = 256
MIN_AMP, MAX_AMP = 0.0, 0.4
NUM_OCTAVES = 2.5
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

def visualise_on_image(image, bboxes, labels, scores, score_thresh, vert_thresh):
    (h, w, d) = image.shape
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > score_thresh and bbox[0] > vert_thresh:
            xmin, ymin = int(bbox[1]*w), int(bbox[0]*h)
            xmax, ymax = int(bbox[3]*w), int(bbox[2]*h)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.circle(image, ((xmin+xmax)//2, (ymin+ymax)//2), 3, (0,255,0), -1)
            cv2.putText(image, f"{label}: {int(score*100)} %", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return image

def overlay(image, overlay_img, ovr_loc):
    image_stacked = np.dstack([image, np.ones(image.shape[:2], dtype="uint8") * 255])
    overlay_size = (
        ovr_loc[2] - ovr_loc[0],
        ovr_loc[3] - ovr_loc[1]
    )
    overlay = cv2.resize(overlay_img, overlay_size, interpolation=cv2.INTER_AREA)
    blend = np.zeros(image_stacked.shape, dtype="uint8")
    blend[ovr_loc[1]:ovr_loc[3], :] = overlay
    final_img = image_stacked.copy()
    final_img = cv2.addWeighted(blend, 0.3, final_img, 1.0, 0, final_img)
    return final_img

###############################################################
cx, cy = 0, 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # float `height`
overlay_img = cv2.imread('keyboard_overlay.png', cv2.IMREAD_UNCHANGED)

freq_lut = (MIN_FREQ * 2**(NUM_OCTAVES * np.arange(W)/W)).astype(np.int)

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
        vert_thresh = 0.5 # ignore bboxes whos ymin coord is less than this
        max_detections = 4

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        scores = detections['detection_scores'][0, :max_detections].numpy()
        bboxes = detections['detection_boxes'][0, :max_detections].numpy()
        labels = detections['detection_classes'][0, :max_detections].numpy().astype(np.int64)
        labels = [category_index[n]['name'] for n in labels]

        bbox_found = False
        for idx, (bbox, score) in enumerate(zip(bboxes, scores)):
            if score >= score_thresh and bbox[0] > vert_thresh:
                cx = (bbox[1]+bbox[3])/2
                cy = (bbox[0]+bbox[2])/2
                bbox_found = True
        
        cy = cy if bbox_found else cy * 0.75

        # Display detections
        visualise_on_image(frame, bboxes, labels, scores, score_thresh, vert_thresh)

        toc = time.time()
        fps = int(1/(toc - tic))
        tic = toc
        cv2.putText(frame, f"FPS: {fps}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        output_img = overlay(frame, overlay_img, [0, 3*H//4, W, H])
        cv2.imshow("Hands Theremin", output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        
        # Get mouse x,y position and convert to freq and amplitude
        freq_in = freq_lut[int(min(cx * W, W))]
        amp_in = cy * (MAX_AMP-MIN_AMP)


print("Exiting ...")
cap.release()
cv2.destroyAllWindows()