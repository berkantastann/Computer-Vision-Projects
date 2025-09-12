import cv2 as cv
import numpy as np
import mediapipe as mp
import argparse

def process_image(img, face_detection):
    H, W = img.shape[:2]

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    out = face_detection.process(img_rgb)


    if out.detections is not None:
        for detection in out.detections:

            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # blur faces

            # img[y1:y1 + h, x1:x1 + w] = cv.blur(img[y1:y1 + h, x1:x1 + w], (30, 30))

            # detect face

            img = cv.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 5)

    return img

args = argparse.ArgumentParser()
args.add_argument("--mode", default="webcam") # Diğer seçenekler -> video, image (path eklemeyi unutmayın!)
args.add_argument("--filePath",default=None)

args = args.parse_args()

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode in ["image"]:

        img = cv.imread(args.filePath)
        img = process_image(img, face_detection)

        cv.imwrite("./output/face_detection.jpg", img)

    elif args.mode in ["video"]:

        cap = cv.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv.VideoWriter("./output/output.mp4",
                                      cv.VideoWriter_fourcc(*"MP4V", ),
                                      25, (frame.shape[1], frame.shape[0]),)
        while ret:

            frame = process_image(frame, face_detection)
            output_video.write(frame)

            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode in ["webcam"]:

        cap = cv.VideoCapture(0)
        ret, frame = cap.read()

        while ret:

            frame = process_image(frame, face_detection)
            cv.imshow("Video", frame)
            cv.waitKey(25)

            ret, frame = cap.read()

        cap.release()