import cv2
import time
import argparse
import numpy as np
import imutils

DEFAULT_VIDEO = "./datasets/volkswagen_pikes_peak_original_cut_3.mp4"
YOLOV3_SPP_WEIGHTS = "./models/yolov3/yolov3-spp.weights"
YOLOV3_SPP_CFG = "./models/yolov3/yolov3-spp.cfg"
YOLOV3_WEIGHTS = "./models/yolov3/yolov3.weights"
YOLOV3_CFG = "./models/yolov3/yolov3.cfg"
CLASSES = open('models/yolov3/coco.names').read().strip().split('\n')
conf = 0.5

colors = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype='uint8')

### YOLO utility funcs


### Driver main
def main():
    ### Argument parser setup
    parser = argparse.ArgumentParser(description='particle filter tracker')
    parser.add_argument('--yolo_model', dest='yolo_model', default="None",
                        help='yolo model to be used by particle filter')
    parser.add_argument('--video', dest='video', default=DEFAULT_VIDEO,
                        help='video to track object')
    parser.add_argument('--gpu', type=bool, default=False)


    args = parser.parse_args()
    yolo_model = args.yolo_model
    video = args.video
    gpu_enable = args.gpu


    ### Load model
    ## YOLO Model
    net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG, YOLOV3_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Use the GPU
    if gpu_enable:  
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


    ### Run the tracker
    cap = cv2.VideoCapture(video)


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()


        # resize the frame, grab the frame dimensions, and convert it to
        # a blob
        (h, w) = frame.shape[:2]
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        outputs = net.forward(ln)

        # combine the 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)


        # loop over each of the layer outputs
        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                if CLASSES[classID] == "car":
                    print(CLASSES[classID])
                    box = output[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                
                    cv2.rectangle(frame, (x, y), (x + width, y+height),
                        (255,255,255), 2)

                # color = [int(c) for c in colors[classID]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                # text = "{}: {:.4f}".format(CLASSES[classID], confidence)
                # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # p0 = int(x - w//2), int(y - h//2)
                # p1 = int(x + w//2), int(y + h//2)
                # boxes.append([*p0, int(w), int(h)])
                # confidences.append(float(confidence))
                # classIDs.append(classID)


        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
            


if __name__ == "__main__":
    main()