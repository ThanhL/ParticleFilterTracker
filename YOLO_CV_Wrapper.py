import cv2
import numpy as np
np.random.seed(seed=501)

class YOLO_CV_Wrapper():
    def __init__(self, yolo_weights, yolo_cfg, yolo_classes, gpu_enabled=False, confidence_thresh=0.5,
                nms_thresh=0.3):
        ### Initialize OpenCV's Darknet
        self.yolo_weights = yolo_weights
        self.yolo_cfg = yolo_cfg
        self.yolo_classes = open(yolo_classes).read().strip().split('\n')
        self.confidence_thresh = confidence_thresh
        self.nms_thresh = nms_thresh

        self.color = np.random.randint(0, 255, size=(len(self.yolo_classes), 3), dtype='uint8')


        ## Darknet creation
        self.net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
        
        ln = self.net.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        ## To GPU or not to GPU
        if gpu_enabled:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def darknet_process_frame(self, frame):
        """
        Take in an opencv frame and runs yolo on the frame. Bounding boxes are drawn
        onto the frame.
    
        Adapted from Adrian Rosebrock's code:
        https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

        """
        ### Extract frame info
        (h, w) = frame.shape[:2]

        ### Create a blob and pass the blob through the network
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        # combine the YOLO's 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)


        # bounding boxes, confidences, class ids
        boxes = []
        confidences = []
        classIDs = []

        ### Iterate through output and apply bounding boxes
        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > self.confidence_thresh:
                box = output[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        ### Apply Non-maxima supression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_thresh, self.nms_thresh)
        

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.color[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.yolo_classes[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)   

        return frame


    def detect_objects(self, frame):
        """
        Take in an opencv frame and runs yolo on the frame. Returns a dictionary of 
        detection containing box location, detection centers, classID and confidence
        """
        ### Extract frame info
        (h, w) = frame.shape[:2]

        ### Create a blob and pass the blob through the network
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        # combine the YOLO's 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)


        # bounding boxes, confidences, class ids
        centers = []
        boxes = []
        confidences = []
        classIDs = []

        nms_detections = []

        ### Iterate through output and apply bounding boxes
        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > self.confidence_thresh:
                box = output[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))


                boxes.append([x, y, int(width), int(height)])
                centers.append([centerX, centerY])
                confidences.append(float(confidence))
                classIDs.append(classID)

        ### Apply Non-maxima supression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_thresh, self.nms_thresh)
        
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                detection = {}
                detection["boxes"] = boxes[i]
                detection["confidence"] = confidences[i]
                detection["classIDs"] = self.yolo_classes[classIDs[i]]
                detection["center"] = centers[i]
                nms_detections.append(detection)

        return nms_detections




