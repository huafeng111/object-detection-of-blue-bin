'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import os
import numpy as np
import cv2
from skimage.measure import label, regionprops

class BinDetector():
    def __init__(self):
        '''
            Initilize your bin detector with the attributes you need,
            e.g., parameters of your classifier
        '''
        # Hyper-parameters.
        self.lr_conf_threshold = 0.6
        val_dir = "data/validation/"

        # Load parameters from classifier.
        folder_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(folder_path, 'model.ckpt.npy')
        self.params_optimal = np.load(model_path)

    # Predict with trained-well parameters.
    # The whole training and validation pipeline can be found in `bin_classification.ipynb`
    def sigmoid(self, x):
        x = x.astype(np.float64)
        results = 1 / (1 + np.exp(-x))
        return results.astype(np.float64)

    def predict_prob(self, X, params):
        return self.sigmoid(X @ params)

    def segment_image(self, img):
        '''
            Obtain a segmented image using a color classifier,
            e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture,
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE

        # Replace this with your own approach
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        w, h, c = img.shape
        flatten_img = img.reshape((w*h, c))
        flatten_img = np.hstack((np.ones((len(flatten_img),1)), flatten_img))

        flatten_preds = self.predict_prob(flatten_img, self.params_optimal)
        flatten_preds = np.array(flatten_preds > self.lr_conf_threshold, np.float)
        mask_img = flatten_preds.reshape((w, h)).astype("uint8")

        # YOUR CODE BEFORE THIS LINE
        ################################################################
        return mask_img

    def get_bounding_boxes(self, img):
        '''
            Find the bounding boxes of the recycling bins
            call other functions in this class if needed

            Inputs:
                img - mask image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE
        w, h, = img.shape
        #Apply compution vision erode and dilation to filter out spike blobs.
        filter_mask = np.ones((5, 5), dtype="uint8")
        erode_preds = cv2.erode(img, filter_mask, iterations=3)
        preds = cv2.dilate(erode_preds, filter_mask, iterations=3)

        hKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,7))
        erodes= cv2.erode(preds, hKernel , iterations=3)
        preds = cv2.dilate(erodes,  hKernel , iterations=3)
        preds=img

        # Get contours and bounding box from segmentation masks.
        contours, _ = cv2.findContours(preds, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for i, contour in enumerate(contours):
            x, y, bw, bh = cv2.boundingRect(contour.astype("float32"))  # X, Y, W, H
            area = cv2.contourArea(contour)
            # Some heuristic rules.
            if area / (w*h) < 0.01:
                continue
            # if bh/bw < 1.0:
            #     continue
            boxes.append([x, y, x+bw, y+bh])

        # YOUR CODE BEFORE THIS LINE
        ################################################################

        return boxes
