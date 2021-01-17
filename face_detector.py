import numpy as np
import cv2
import matplotlib.pyplot as plt

frontal_face_cascade = cv2.CascadeClassifier('facedetectioncascades/haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier('facedetectioncascades/haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('facedetectioncascades/haarcascade_eye.xml')

class FaceDetector():

    def rescale(img):
        factor = 1300 / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*factor), int(img.shape[0]*factor)))
        return img
    
    def detect_face(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = int(min(img.shape[1], img.shape[0])/20)
        faces = frontal_face_cascade.detectMultiScale(gray, 1.1, 4, minSize = (size, size))
        if len(faces) > 0:
            return faces
        else:
            print("No faces detected in image.")
       
    def draw_rectangles(faces, img):
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        plt.axis("off")
        plt.imshow(img)

    def crop(sub_images, img):
        cropped = []
        for (x, y, w, h) in sub_images:
            sub_image = img[y:y+h, x:x+w]
            cropped.append(sub_image)
        return cropped
    
    def cropped_faces(img):
        img = FaceDetector.rescale(img)
        faces = FaceDetector.detect_face(img)
        faces = FaceDetector.non_max_suppression_slow(faces, 0.1)
        cropped = FaceDetector.crop(faces, img)
        cropped = [face for face in cropped if FaceDetector.is_valid_face(face)]
        return cropped
            
    def show_faces(img):
        face_imgs = FaceDetector.cropped_faces(img)
        counter = 1
        fig = plt.figure(figsize=(20, 20))
        for img in face_imgs:
            ax = fig.add_subplot(1, len(face_imgs), counter)
            ax.axis("off")
            plt.imshow(img)
            counter += 1
        plt.show()
        
    def detect_eyes(face):
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.05, 1)
        return eyes
    
    def is_valid_face(face):
        eyes = FaceDetector.detect_eyes(face)
        return len(eyes) > 0
    
    def non_max_suppression_slow(boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
        
        # convert to corner coordinates
        boxes = FaceDetector.to_corner_coord(boxes)
        
        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(x1)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list, add the index
            # value to the list of picked indexes, then initialize
            # the suppression list (i.e. indexes that will be deleted)
            # using the last index
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]

            # loop over all indexes in the indexes list
            for pos in range(0, last):
                # grab the current index
                j = idxs[pos]

                # find the largest (x, y) coordinates for the start of
                # the bounding box and the smallest (x, y) coordinates
                # for the end of the bounding box
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])

                # compute the width and height of the bounding box
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                # compute the ratio of overlap between the computed
                # bounding box and the bounding box in the area list
                overlap = float(w * h) / area[j]

                # if there is sufficient overlap, suppress the
                # current bounding box
                if overlap > overlapThresh:
                    suppress.append(pos)

            # delete all indexes from the index list that are in the
            # suppression list
            idxs = np.delete(idxs, suppress)

        # return only the bounding boxes that were picked
        return FaceDetector.to_top_left(boxes[pick])
    
    def to_corner_coord(rects):
        for i in range(len(rects)):
            rects[i, 2] += rects[i, 0]
            rects[i, 3] += rects[i, 1]
        return rects
    
    def to_top_left(rects):
        for i in range(len(rects)):
            rects[i, 2] -= rects[i, 0]
            rects[i, 3] -= rects[i, 1]
        return rects
            