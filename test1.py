# # import the necessary packages
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# from threading import Thread
# import sys
# import cv2
# from multiprocessing import Queue, Pool
# import time
# import numpy as np 
# import tensorflow as tf

# from utils import label_map_util
# from utils import visualization_utils

# class WebcamVideoStream:
#     def __init__(self, src=0):
#         # initialize the video camera stream and read the first frame
#         # from the stream
#         self.stream = cv2.VideoCapture(src)
#         (self.grabbed, self.frame) = self.stream.read()

#         # initialize the variable used to indicate if the thread should
#         # be stopped
#         self.stopped = False

#     def start(self):
#         # start the thread to read frames from the video stream
#         Thread(target=self.update, args=()).start()
#         return self

#     def update(self):
#         # keep looping infinitely until the thread is stopped
#         while True:
#             # if the thread indicator variable is set, stop the thread
#             if self.stopped:
#                 return

#             # otherwise, read the next frame from the stream
#             (self.grabbed, self.frame) = self.stream.read()

#     def read(self):
#         # return the frame most recently read
#         return self.frame

#     def size(self):
#         # return size of the capture device
#         return self.stream.get(3), self.stream.get(4)

#     def stop(self):
#         # indicate that the thread should be stopped
#         self.stopped = True

# def worker(input_q, output_q):


#     while True:

#         image_np = input_q.get()
#         if (image_np is not None):
#             # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
#             # while scores contains the confidence for each of these boxes.
#             # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
#             pass
#         output_q.put(image_np)


# if __name__=='__main__':
#     input_q = Queue(maxsize=100)
#     output_q = Queue(maxsize=100)
#     pool = Pool(None, worker, (input_q, output_q))

#     MODEL_NAME = 'ssd_mobilenet_v2_fpn_320'
#     PATH_TO_SAVED_MODEL = os.path.join(r'C:\Users\ap5\Documents\hands\detect_hands', 'model_data', MODEL_NAME, 'saved_model')
#     # List of the strings that is used to add correct label for each box.
#     PATH_TO_LABELS = os.path.join(r'C:\Users\ap5\Documents\hands\detect_hands', 'model_data', MODEL_NAME, 'label_map.pbtxt')
#     NUM_CLASSES = 1

#     print("Loading saved model ...")
#     detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
#     print("Model Loaded!")

#     # Load label map and obtain class names and ids
#     label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#     categories = label_map_util.convert_label_map_to_categories(
#         label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#     category_index = label_map_util.create_category_index(categories)    

#     cap = WebcamVideoStream().start()
#     # Allow some time for the thread to collect video frames
#     time.sleep(1.0)

#     tic = time.time()
#     # loop over frames from the video file stream
#     while True:
#         # grab the frame from the threaded video file stream, resize
#         # it, and convert it to grayscale (while still retaining 3
#         # channels)
#         frame = cap.read()
#         input_q.put(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))

#         image_np = output_q.get()


#         # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
#         input_tensor = tf.convert_to_tensor(image_np)
#         # The model expects a batch of images, so add an axis with `tf.newaxis`.
#         input_tensor = input_tensor[tf.newaxis, ...]

#         detections = detect_fn(input_tensor)

#         # All outputs are batches tensors.
#         # Convert to numpy arrays, and take index [0] to remove the batch dimension.
#         # We're only interested in the first num_detections.
#         num_detections = int(detections.pop('num_detections'))
#         max_detections = 4
#         detections = {key: value[0, :max_detections].numpy()
#                     for key, value in detections.items()}
#         detections['num_detections'] = num_detections
#         # detection_classes should be ints.
#         detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

#         image_np_with_detections = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
#         visualization_utils.visualize_boxes_and_labels_on_image_array(
#                 image_np_with_detections,
#                 detections['detection_boxes'],
#                 detections['detection_classes'],
#                 detections['detection_scores'],
#                 category_index,
#                 use_normalized_coordinates=True,
#                 max_boxes_to_draw=200,
#                 min_score_thresh=.30)

#         toc = time.time()
#         fps = int(1/(toc - tic))
#         tic = toc
#         cv2.putText(image_np_with_detections, f"FPS: {fps}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
#         cv2.imshow("img", image_np_with_detections)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     pool.terminate()
#     cap.stop()
#     cv2.destroyAllWindows()
def overlay(image, overlay_img, ovr_loc):
    image = np.dstack([image, np.ones(image.shape[:2], dtype="uint8") * 255])
    overlay_size = (
        ovr_loc[2] - ovr_loc[0],
        ovr_loc[3] - ovr_loc[1]
    )
    overlay = cv2.resize(overlay_img, overlay_size, interpolation=cv2.INTER_AREA)
    blend = np.zeros(image.shape, dtype="uint8")
    blend[ovr_loc[1]:ovr_loc[3], :] = overlay
    final = cv2.addWeighted(blend, 0.3, image, 1.0, 0, image)
    return final



import cv2
import numpy as np 
cap = cv2.VideoCapture(0)
ret, img = cap.read()
print(img.shape)
(H, W, D) = img.shape
cv2.imshow("img", img)
cv2.waitKey(0)

final = overlay(img, 'keyboard.png', [0, 3*H//4, W, H])

cv2.imshow("combined", final)
cv2.waitKey(0)

cv2.destroyAllWindows()
