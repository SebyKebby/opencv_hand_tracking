
#task : Implement opencv2 on the camera feed : Done
#           Fix the orange screen bug : Done ( Caused by droidcam)
#       Implement mediapipe (done)
#       Implement handtracking(done)
#       Implement overlay 
#       Convert handtracked motion to keyboard input
#       Do the deed
import numpy as np
import cv2 as cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

#drawing 
def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
        #extracting handlandmark results

    #draw
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
    frame,
    hand_landmarks_proto,
    solutions.hands.HAND_CONNECTIONS,
    solutions.drawing_styles.get_default_hand_landmarks_style(),
    solutions.drawing_styles.get_default_hand_connections_style())
  return annotated_image


#create Handmarker in livestream
def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

#mp options
options = vision.HandLandmarkerOptions(base_options=BaseOptions('F:\wababjack\image_recognition_pressing_script\opencv_hand_tracking\hand_landmarker.task'),
                                       running_mode=VisionRunningMode.LIVE_STREAM,
                                       num_hands=2,
                                       result_callback = print_result)
detector = vision.HandLandmarker.create_from_options(options)

#initiate
with HandLandmarker.create_from_options(options) as landmarker:
    #initiate opencv feed
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    calc_timestamps = [0.0]
    #get timestampe
    while(cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
        else:
            break

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_np = np.array(frame) 
        timestamp = calc_timestamps
        mp_image =  mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_np)
        frame = mp_image.numpy_view()
        result = landmarker.detect_async(mp_image, timestamp)
        annotated_image = draw_landmarks_on_image(frame.numpy_view(), result)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

