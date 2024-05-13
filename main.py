import warnings
import argparse
import cv2

from nets.nn import *
from utils.util import norm_crop_image

warnings.filterwarnings("ignore")

detection = FaceDetector('./weights/detection.onnx')
recognition = FaceRecognition('./weights/recognition_r50.onnx')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', default=0.35, help='image file paths')
    parser.add_argument('--ref_image', default='demo/Mask.jpg', help='reference image file paths')

    args = parser.parse_args()
    font = cv2.FONT_HERSHEY_DUPLEX

    ref_image = cv2.imread(args.ref_image)
    name_id = args.ref_image.split('/')[-1].split('.')[0]

    _, kpt = detection.detect(ref_image, input_size=(640, 640))
    kpt = kpt[:1][0]
    face = norm_crop_image(ref_image, kpt)

    stream = cv2.VideoCapture(0)

    while True:
        success, frame = stream.read()
        if success:
            boxes, kpt = detection.detect(frame, input_size=(640, 640))
            if kpt.size == 0:
                continue

            kpt = kpt[:1][0]
            test_face = norm_crop_image(frame, kpt)

            x1 = int(boxes[0][0])
            y1 = int(boxes[0][1])
            x2 = int(boxes[0][2])
            y2 = int(boxes[0][3])

            vector1 = recognition(face)[0].flatten()
            vector2 = recognition(test_face)[0].flatten()

            score = vector1 @ vector2
            print(score)
            if score > 0.35:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)
                cv2.putText(frame, name_id, (x1 + 6, y1 - 6), font, 1.0, (255, 0, 255), 1)

                cv2.line(frame, (int(x1), int(y1)), (int(x1 + 15), int(y1)), (255, 0, 255), 3)
                cv2.line(frame, (int(x1), int(y1)), (int(x1), int(y1 + 15)), (255, 0, 255), 3)

                cv2.line(frame, (int(x2), int(y2)), (int(x2 - 15), int(y2)), (255, 0, 255), 3)
                cv2.line(frame, (int(x2), int(y2)), (int(x2), int(y2 - 15)), (255, 0, 255), 3)

                cv2.line(frame, (int(x2 - 15), int(y1)), (int(x2), int(y1)), (255, 0, 255), 3)
                cv2.line(frame, (int(x2), int(y1)), (int(x2), int(y1 + 15)), (255, 0, 255), 3)

                cv2.line(frame, (int(x1), int(y2 - 15)), (int(x1), int(y2)), (255, 0, 255), 3)
                cv2.line(frame, (int(x1), int(y2)), (int(x1 + 15), int(y2)), (255, 0, 255), 3)
            else:
                name = 'Unknown'
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv2.putText(frame, name, (x1 + 6, y1 - 6), font, 1.0, (0, 0, 255), 1)

                cv2.line(frame, (int(x1), int(y1)), (int(x1 + 15), int(y1)), (0, 0, 255), 3)
                cv2.line(frame, (int(x1), int(y1)), (int(x1), int(y1 + 15)), (0, 0, 255), 3)

                cv2.line(frame, (int(x2), int(y2)), (int(x2 - 15), int(y2)), (0, 0, 255), 3)
                cv2.line(frame, (int(x2), int(y2)), (int(x2), int(y2 - 15)), (0, 0, 255), 3)

                cv2.line(frame, (int(x2 - 15), int(y1)), (int(x2), int(y1)), (0, 0, 255), 3)
                cv2.line(frame, (int(x2), int(y1)), (int(x2), int(y1 + 15)), (0, 0, 255), 3)

                cv2.line(frame, (int(x1), int(y2 - 15)), (int(x1), int(y2)), (0, 0, 255), 3)
                cv2.line(frame, (int(x1), int(y2)), (int(x1 + 15), int(y2)), (0, 0, 255), 3)

            cv2.imshow('Frame', frame)
            key = cv2.waitKey(20)

            if key == ord('q'):
                break
        else:
            break


if __name__ == '__main__':
    main()
