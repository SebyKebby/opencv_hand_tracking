#task : Implement opencv2 on the camera feed : Done
#           Fix the orange screen bug : Done ( Caused by droidcam)
#       Implement mediapipe (done)
#       Implement handtracking(done)
#       Implement overlay (done)
#       Create a system to interpret hand motion (done)
#       Convert handtracked motion to keyboard input (done)
#       Do the deed

import pyautogui
import cv2
import mediapipe as mp
import time

def count_fingers(lst):
    cnt = 0

    thresh = (lst.landmark[0].y*100 - lst.landmark[9].y*100) / 2

    if (lst.landmark[5].y*100 - lst.landmark[8].y*100) > thresh :
            cnt +=1
    if (lst.landmark[9].y*100 - lst.landmark[12].y*100) > thresh :
            cnt +=1
    if (lst.landmark[13].y*100 - lst.landmark[16].y*100) > thresh :
            cnt +=1
    if (lst.landmark[17].y*100 - lst.landmark[20].y*100) > thresh :
        cnt +=1
    if (lst.landmark[4].y*100 - lst.landmark[5].y*100) > 3 :
          cnt +=1
    return cnt

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

prev = -1
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
            
        hand_keyPoints = results.multi_hand_landmarks[0]

        print(count_fingers(hand_keyPoints))

        cnt = count_fingers(hand_keyPoints)
        if not(prev == cnt) :
            if (cnt == 4):
                pyautogui.click(button = 'left')
            elif (cnt == 3):
                pyautogui.click(button = 'right')
            prev = cnt

        mpDraw.draw_landmarks(img, hand_keyPoints, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)