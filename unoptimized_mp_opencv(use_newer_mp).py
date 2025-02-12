
#task : Implement opencv2 on the camera feed : Done
#           Fix the orange screen bug : Done ( Caused by droidcam)
#       Implement mediapipe (done)
#       Implement handtracking(done)
#       Implement overlay (done)
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
latest_result = None #global var for latest_result

#drawing 
def draw_landmarks_on_image(rgb_image, detection_result):
    if detection_result is None or not hasattr(detection_result, 'hand_landmarks'):
        return rgb_image
    
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        
        # Convert landmarks to proto format
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in hand_landmarks
        ])
        
        # Draw landmarks
        solutions.drawing_utils.draw_landmarks(
            rgb_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
        
        # Optionally draw handedness (left/right hand)
        if handedness_list:
            handedness = handedness_list[idx]
            text_x = int(min(hand_landmarks[0].x, hand_landmarks[1].x) * rgb_image.shape[1])
            text_y = int(min(hand_landmarks[0].y, hand_landmarks[1].y) * rgb_image.shape[0]) - 10
            cv2.putText(rgb_image, f"{handedness[0].category_name}",
                      (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                      0.7, HANDEDNESS_TEXT_COLOR, 2)
    
    return rgb_image


#create Handmarker in livestream
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

#mp options
options = vision.HandLandmarkerOptions(base_options=BaseOptions('hand_landmarker.task'),
                                       running_mode=VisionRunningMode.LIVE_STREAM,
                                       num_hands=2,
                                       result_callback = print_result)

#initiate
with HandLandmarker.create_from_options(options) as landmarker:
    #initiate opencv feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

        #fps for timestamp
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp += int(1000 / fps)  # convert to milliseconds / increment

        mp_image =  mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # detect hands
        landmarker.detect_async(mp_image, timestamp)
        
        annotated_image = draw_landmarks_on_image(frame, latest_result)
        cv2.imshow("frame", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


