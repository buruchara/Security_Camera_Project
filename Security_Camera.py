import cv2
import time
import datetime

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

recoding = False
detection_stop_time = None
timer_started = False
Seconds_To_Record_After_Detection = 10

frame_size = (int(cam.get(3)), int(cam.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    _, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    bodies = body_cascade.detectMultiScale(gray, 1.1, 6)

    if len(faces) + len(bodies) > 0:
        if recoding:
            timer_started = False
        else:
            recoding = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H%M-%S")
            output = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Started recoding!")
    elif recoding:
        if timer_started:
            if time.time() - detection_stop_time >= Seconds_To_Record_After_Detection:
                recoding = False
                output.release()
                print('Stop Recoding!')
            else:
                timer_started = True
                detection_stop_time = time.time()

    if recoding:
        output.write(frame)

    # for (x, y, width, height) in faces:
    #     cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)



    cv2.imshow("Camera", frame)


    if cv2.waitKey(1) ==ord('q'):
        break

output.release()
cam.release()
cv2.destroyAllWindows()
