import handDetection
import cv2
import time
import mouseController as controller
from pyautogui import size
import threading

# Global Constants
SMOOTH_FACTOR = 0.3
DEADZONE_RANGE = 0.005

def euclidean_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def low_pass_filter(x, y, locOffsetX, locOffsetY, prev_x, prev_y):
    # Smooth cursor movement using a low-pass filter
    new_x = (1 - SMOOTH_FACTOR) * prev_x + SMOOTH_FACTOR * (x + locOffsetX)
    new_y = (1 - SMOOTH_FACTOR) * prev_y + SMOOTH_FACTOR * (y + locOffsetY)
    return new_x, new_y

def deadzone(current_x, current_y, new_x, new_y, screen_size):
    # Apply a dead zone to avoid small, jittery movements
    dx = abs(new_x - current_x)
    dy = abs(new_y - current_y)
    deadzone_x = screen_size[0] * DEADZONE_RANGE
    deadzone_y = screen_size[1] * DEADZONE_RANGE

    if dx < deadzone_x:
        new_x = current_x
    if dy < deadzone_y:
        new_y = current_y

    return new_x, new_y

def smooth_scroll(amount, duration, stop_event):
    steps = 100  # Number of steps for the scroll
    step_amount = amount / steps
    step_duration = duration / steps
    
    while not stop_event.is_set():
        controller.scroll_mouse(step_amount)
        time.sleep(step_duration)
stop_event = threading.Event()

def main():

    # Screen variables
    screen_size = size()
    cam_res_width = 1280
    cam_res_height = 720
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_res_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_res_height)

    # Time variables
    prev_time = time.time()
    prev_time_fist = time.time()
    prev_time_four = time.time()
    prev_time_Yo = time.time()
    prev_time_three = time.time()
    prev_time_scroll = time.time()
    prev_time_scroll_state = time.time()

    # Constant or Initialize variables
    pTime = 0
    cTime = 0
    prev_x, prev_y = 0, 0  # Initialize previous cursor position
    wait_time_stop = 3
    wait_time_gesture = 1
    offsetX = 200
    offsetY = 200
    scrollSpeed = 2
    scrollDistance = 100
    offsetAllowed = False
    detectionOn = False
    isScrolling = False
    middleFigY = 0
    scroll_threshold = 100

    # Detection Class initialization
    detector = handDetection.HandDetector()

    # Main loop
    while True:

        # Capture frame-by-frame
        success, img = cap.read()
        img = cv2.flip(img, 1)
        if img is not None:
            img = cv2.resize(img, screen_size)
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        h, w, c = img.shape
        factor = cam_res_width/h
        
        # Get finger positions
        if detector.getPositions(img) is not None:
            idxFigX = detector.getPositions(img)['idx_tip'][0]*factor
            idxFigY = detector.getPositions(img)['idx_tip'][1]*factor
            idxFigZ = detector.getPositions(img)['idx_tip'][2]*factor
            thumbFigX = detector.getPositions(img)['thumb_tip'][0]*factor
            thumbFigY = detector.getPositions(img)['thumb_tip'][1]*factor
            midFigX = detector.getPositions(img)['mid_tip'][0]*factor
            midFigY = detector.getPositions(img)['mid_tip'][1]*factor 
            wristX = detector.getPositions(img)['wrist'][0]*factor
            wristY = detector.getPositions(img)['wrist'][1]*factor
            ringFigX = detector.getPositions(img)['ring_tip'][0]*factor
            ringFigY = detector.getPositions(img)['ring_tip'][1]*factor 
            pinkyFigX = detector.getPositions(img)['pinky_tip'][0]*factor
            pinkyFigY = detector.getPositions(img)['pinky_tip'][1]*factor
            midDipX = detector.getPositions(img)['mid_dip'][0]*factor
            midDipY = detector.getPositions(img)['mid_dip'][1]*factor
            ringDipX = detector.getPositions(img)['ring_dip'][0]*factor
            ringDipY = detector.getPositions(img)['ring_dip'][1]*factor
            pinkyDipX = detector.getPositions(img)['pinky_dip'][0]*factor
            pinkyDipY = detector.getPositions(img)['pinky_dip'][1]*factor
            idxDipX = detector.getPositions(img)['idx_dip'][0]*factor
            idxDipY = detector.getPositions(img)['idx_dip'][1]*factor
            thumbDipX = detector.getPositions(img)['thumb_dip'][0]*factor
            thumbDipY = detector.getPositions(img)['thumb_dip'][1]*factor

            # Geusture Detection

            # Fist Detection
            if(euclidean_distance(idxFigX, idxFigY, wristX, wristY) < euclidean_distance(idxDipX, idxDipY, wristX, wristY)
               and euclidean_distance(midFigX, midFigY, wristX, wristY) < euclidean_distance(midDipX, midDipY, wristX, wristY)
               and euclidean_distance(ringFigX, ringFigY, wristX, wristY) < euclidean_distance(ringDipX, ringDipY, wristX, wristY)
               and euclidean_distance(pinkyFigX, pinkyFigY, wristX, wristY) < euclidean_distance(pinkyDipX, pinkyDipY, wristX, wristY)
               and abs(thumbFigX - wristX) < abs(thumbDipX - wristX)):
                curr_time_fist = time.time()
                time_diff_fist = curr_time_fist-prev_time_fist
                # cv2.putText(img, f"Time: {time_diff_fist}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if time_diff_fist >= wait_time_stop:
                    detectionOn = False
                    prev_time_fist = curr_time_fist
            else:
                prev_time_fist = time.time()

            # Four Detection
            if(euclidean_distance(idxFigX, idxFigY, wristX, wristY) > euclidean_distance(idxDipX, idxDipY, wristX, wristY)
               and euclidean_distance(midFigX, midFigY, wristX, wristY) > euclidean_distance(midDipX, midDipY, wristX, wristY)
               and euclidean_distance(ringFigX, ringFigY, wristX, wristY) > euclidean_distance(ringDipX, ringDipY, wristX, wristY)
               and euclidean_distance(pinkyFigX, pinkyFigY, wristX, wristY) > euclidean_distance(pinkyDipX, pinkyDipY, wristX, wristY)
               and abs(thumbFigX - wristX) < abs(thumbDipX - wristX)):
                curr_time_four = time.time()
                time_diff_four = curr_time_four-prev_time_four
                # cv2.putText(img, f"Time: {time_diff_four}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if time_diff_four >= wait_time_stop:
                    detectionOn = True
                    prev_time_four = curr_time_four
            else:
                prev_time_four = time.time()

            # Yoo Geusture Detection
            if (euclidean_distance(midFigX, midFigY, wristX, wristY) < euclidean_distance(midDipX, midDipY, wristX, wristY) 
                and euclidean_distance(ringFigX, ringFigY, wristX, wristY) < euclidean_distance(ringDipX, ringDipY, wristX, wristY)
                and euclidean_distance(pinkyFigX, pinkyFigY, wristX, wristY) > euclidean_distance(pinkyDipX, pinkyDipY, wristX, wristY)
                and euclidean_distance(idxFigX, idxFigY, wristX, wristY) > euclidean_distance(idxDipX, idxDipY, wristX, wristY)):
                curr_time_Yo = time.time()
                time_diff_Yo = curr_time_Yo-prev_time_Yo
                # cv2.putText(img, f"Time: {time_diff_Yo}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if time_diff_Yo >= wait_time_gesture:
                    offsetAllowed = True
                    prev_time_Yo = curr_time_Yo
            else:
                prev_time_Yo = time.time()

            # Index, Middle and Thumb Gesture
            if (euclidean_distance(pinkyFigX, pinkyFigY, wristX, wristY) < euclidean_distance(pinkyDipX, pinkyDipY, wristX, wristY)
                and euclidean_distance(ringFigX, ringFigY, wristX, wristY) < euclidean_distance(ringDipX, ringDipY, wristX, wristY)
                and euclidean_distance(midFigX, midFigY, wristX, wristY) > euclidean_distance(midDipX, midDipY, wristX, wristY)
                and euclidean_distance(idxFigX, idxFigY, wristX, wristY) > euclidean_distance(idxDipX, idxDipY, wristX, wristY)
                and abs(thumbFigX-wristX) > abs(thumbDipX-wristX)):
                curr_time_three = time.time()
                time_diff_three = curr_time_three-prev_time_three
                # cv2.putText(img, f"Time: {time_diff_three}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if time_diff_three >= wait_time_gesture:
                    offsetAllowed = False
                    prev_time_three = curr_time_three
            else:
                prev_time_three = time.time()

            if offsetAllowed:
                if idxFigX < screen_size[0]/2:
                    _offsetX = -offsetX
                else:
                    _offsetX = offsetX
                if idxFigY < screen_size[1]/2:
                    _offsetY = -offsetY
                else:
                    _offsetY = offsetY + 200
            else:
                _offsetX = 50
                _offsetY = 50

            if detectionOn:
                # Cursor Movement
                # Smooth movement with low-pass filter
                new_x, new_y = low_pass_filter(idxFigX + _offsetX, idxFigY + _offsetY, 0, 0, prev_x, prev_y)
                # Apply deadzone
                new_x, new_y = deadzone(prev_x, prev_y, new_x, new_y, screen_size)
                prev_x, prev_y = new_x, new_y
                # Move the mouse to the new position
                controller.moveMouseTo(int(new_x), int(new_y))



                # Click System
                if (euclidean_distance(pinkyFigX, pinkyFigY, wristX, wristY) < euclidean_distance(pinkyDipX, pinkyDipY, wristX, wristY)
                    and euclidean_distance(ringFigX, ringFigY, wristX, wristY) < euclidean_distance(ringDipX, ringDipY, wristX, wristY)
                    and euclidean_distance(midFigX, midFigY, wristX, wristY) < euclidean_distance(midDipX, midDipY, wristX, wristY)):
                    # print(f"[ X: {max(0, idxFigX)}, Y: {max(0, idxFigY)} ]")
                    # print(f"[ X: {thumbFigX}, Y: {thumbFigY} ]")
                    distThumbIdx = euclidean_distance(
                        thumbFigX, thumbFigY, idxFigX, idxFigY)
                    # print(distThumbIdx)
                    curr_time = time.time()
                    time_diff = curr_time-prev_time
                    # print(distThumbIdx)
                    if distThumbIdx < 50:
                        # cv2.putText(img, f"Click Time: {time_diff}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                        if time_diff >= .5:
                                controller.leftClick()
                                prev_time = curr_time



                # Scrolling System
                if (euclidean_distance(pinkyFigX, pinkyFigY, wristX, wristY) < euclidean_distance(pinkyDipX, pinkyDipY, wristX, wristY)
                    and euclidean_distance(ringFigX, ringFigY, wristX, wristY) < euclidean_distance(ringDipX, ringDipY, wristX, wristY)
                    and euclidean_distance(midFigX, midFigY, wristX, wristY) > euclidean_distance(midDipX, midDipY, wristX, wristY)
                    and euclidean_distance(idxFigX, idxFigY, wristX, wristY) > euclidean_distance(idxDipX, idxDipY, wristX, wristY)):
                    curr_time_scroll = time.time()
                    time_diff_scroll = curr_time_scroll-prev_time_scroll
                    if time_diff_scroll >= wait_time_gesture:
                        if not isScrolling:
                            isScrolling = True
                            stop_event.clear()
                            movement = middleFigY - midFigY
                            if movement > scroll_threshold:
                                scrollDistance = -scrollDistance

                            elif movement < -scroll_threshold:
                                scrollDistance = scrollDistance

                            # Start scroll thread
                            scroll_thread = threading.Thread(target=smooth_scroll, args=(scrollDistance, scrollSpeed, stop_event))
                            scroll_thread.start()

                            prev_time_scroll = curr_time_scroll
                            middleFigY = midFigY
                else:
                    if isScrolling:
                        isScrolling = False
                        stop_event.set()
                        scroll_thread.join()
                        prev_time_Yo = time.time()
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        # cv2.putText(img, f"FPS: {int(fps)}", (w-200, 70),
        #             cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        # cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()