import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import mediapipe as mp
import cv2
from google.protobuf.json_format import MessageToDict 


class HandDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=.8, trackCon=0.8) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def getPositions(self, img):
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.idx_tip_x, self.idx_tip_y, self.idx_tip_z = int(hand_landmarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP].x*w), int(
                    hand_landmarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP].y*h), int(hand_landmarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP].z)
                self.thumb_tip_x, self.thumb_tip_y = int(hand_landmarks.landmark[self.mpHands.HandLandmark.THUMB_TIP].x*w), int(
                    hand_landmarks.landmark[self.mpHands.HandLandmark.THUMB_TIP].y*h)
                self.mid_tip_x, self.mid_tip_y = int(hand_landmarks.landmark[self.mpHands.HandLandmark.MIDDLE_FINGER_TIP].x*w), int(
                    hand_landmarks.landmark[self.mpHands.HandLandmark.MIDDLE_FINGER_TIP].y*h)
                self.wrist_x, self.wrist_y = int(hand_landmarks.landmark[self.mpHands.HandLandmark.WRIST].x*w), int(
                    hand_landmarks.landmark[self.mpHands.HandLandmark.WRIST].y*h)
                self.ring_tip_x, self.ring_tip_y = int(hand_landmarks.landmark[self.mpHands.HandLandmark.RING_FINGER_TIP].x*w), int(
                    hand_landmarks.landmark[self.mpHands.HandLandmark.RING_FINGER_TIP].y*h)
                self.pinky_tip_x, self.pinky_tip_y = int(hand_landmarks.landmark[self.mpHands.HandLandmark.PINKY_TIP].x*w), int(
                    hand_landmarks.landmark[self.mpHands.HandLandmark.PINKY_TIP].y*h)
                self.mid_dip_x, self.mid_dip_y = int(hand_landmarks.landmark[self.mpHands.HandLandmark.MIDDLE_FINGER_DIP].x*w), int(
                    hand_landmarks.landmark[self.mpHands.HandLandmark.MIDDLE_FINGER_DIP].y*h)
                self.ring_dip_x, self.ring_dip_y = int(hand_landmarks.landmark[self.mpHands.HandLandmark.RING_FINGER_DIP].x*w), int(
                    hand_landmarks.landmark[self.mpHands.HandLandmark.RING_FINGER_DIP].y*h)
                self.pinky_dip_x, self.pinky_dip_y = int(hand_landmarks.landmark[self.mpHands.HandLandmark.PINKY_DIP].x*w), int(
                    hand_landmarks.landmark[self.mpHands.HandLandmark.PINKY_DIP].y*h)
                self.idx_dip_x, self.idx_dip_y = int(hand_landmarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_DIP].x*w), int(
                    hand_landmarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_DIP].y*h)
                self.thumb_dip_x, self.thumb_dip_y = int(hand_landmarks.landmark[self.mpHands.HandLandmark.THUMB_IP].x*w), int(
                    hand_landmarks.landmark[self.mpHands.HandLandmark.THUMB_IP].y*h)
                return {'idx_tip': [self.idx_tip_x, self.idx_tip_y, self.idx_tip_z],
                        'thumb_tip':  [self.thumb_tip_x, self.thumb_tip_y],
                        'mid_tip':  [self.mid_tip_x, self.mid_tip_y],
                        'wrist': [self.wrist_x, self.wrist_y],
                        'ring_tip': [self.ring_tip_x, self.ring_tip_y],
                        'pinky_tip': [self.pinky_tip_x, self.pinky_tip_y],
                        'mid_dip': [self.mid_dip_x, self.mid_dip_y],
                        'ring_dip': [self.ring_dip_x, self.ring_dip_y],
                        'pinky_dip': [self.pinky_dip_x, self.pinky_dip_y],
                        'idx_dip': [self.idx_dip_x, self.idx_dip_y],
                        'thumb_dip': [self.thumb_dip_x, self.thumb_dip_y]}

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)
        return lmList
    
    def getResults(self):
        return self.results
