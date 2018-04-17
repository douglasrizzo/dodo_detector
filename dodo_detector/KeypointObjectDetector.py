#!/usr/bin/env python

import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
from imutils.video import WebcamVideoStream
from dodo_detector.ObjectDetector import ObjectDetector
from pprint import PrettyPrinter

class KeypointObjectDetector(ObjectDetector):
    def __init__(
            self,
            database_path,
            detector_type='SIFT',
            log_dir='/tmp',
    ):
        self.first_frame = None
        self.current_frame = 0

        self.log_dir = log_dir

        # how many frames to ignore during object detection
        self.frame_skip = 5

        # how many frames an object must appear in order to be considered present in the scene
        self.min_frames = 5

        # how many frames the object detection node will analyze before giving its judgement
        self.max_frames = 20

        # get the directory where object textures are stored
        self.database_path = database_path

        # minimum number of features for a KNN match to consider that an object has been found
        self.min_points = 200

        # create the detector
        if detector_type == 'SIFT':
            self.detector = cv2.xfeatures2d.SIFT_create()
        elif detector_type == 'SURF':
            self.detector = cv2.xfeatures2d.SURF_create()

        # get which OpenCV feature matcher the user wants from the configuration files
        self.matcher_type = 'BF'
        if self.matcher_type == 'BF':
            self.matcher = cv2.BFMatcher()
        elif self.matcher_type == 'FLANN':
            flann_index_kdtree = 0
            index_params = dict(algorithm=flann_index_kdtree, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        self.previous_detected_objects = []
        self.consensus_counter = 0
        self.min_consensus = 5

        # store object classes in a list
        # each directory in the object database corresponds to a class
        self.objects = [os.path.basename(d) for d in os.listdir(self.database_path) if d != 'IGNORE']

        # minimum object dimensions in pixels
        self.min_object_height = 10
        self.min_object_width = 10
        self.min_object_area = self.min_object_height * self.min_object_width

        # initialize the frame counter for each object class at 0
        self.object_counters = {}
        for ob in self.objects:
            self.object_counters[ob] = 0

        # load features for each texture and store the image,
        # its keypoints and corresponding descriptor
        self.object_features = [
            self._load_features(obj) for obj in self.objects
        ]

        cv2.namedWindow("Object Recognition", 1)

    def _load_features(self, obj, return_image=True):
        # Initialize detector
        print('Analyzing image files in \'' + obj + '\'...')

        img_files = [
            join(self.database_path + object_name + '/', f) for f in listdir(self.database_path + object_name + '/')
            if isfile(join(self.database_path + object_name + '/', f))
        ]
        print('Found ' + str(len(img_files)) + ' images...')

        # extract the keypoints from all images in the database
        features = []
        for counter, img_file in enumerate(img_files):
            print('    Extracting keypoints and descriptors from image ' +
                  str(counter) + ': ' + img_file)
            img = cv2.imread(img_file)

            # scaling_factor = 640 / img.shape[0]
            if img.shape[0] > 1000:
                img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

            # find keypoints and descriptors with the selected feature detector
            kp, des = self.detector.detectAndCompute(img, None)

            print('    Extracted ' + str(len(kp)) + ' keypoints!')
            if return_image:
                features.append([img, kp, des])
            else:
                features.append([kp, des])

        return obj, features

    def _detect_object(self, obj_features, scene, coordinates=None):
        name, img_features = obj_features
        scene_img, scene_kp, scene_des = scene

        # print('-' * 80)
        # print('Searching for object \'' + name + '\'...')

        for img_counter, img_feature in enumerate(img_features):
            obj_img, kp, des = img_feature

            if des is not None and len(des) > 0 and scene_des is not None and len(scene_des) > 0:
                matches = self.matcher.knnMatch(des, scene_des, k=2)

                good = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < 0.7 * n.distance:
                            good.append(m)

                # goodness_ratio = float(len(good)) / float(len(kp))

                # print('Image ' + str(img_counter + 1) +
                #       '    Keypoints: ' + str(len(kp)) +
                #       '    Good matches: ' + str(len(good)) +
                #       '    Ratio: ' + str(goodness_ratio))

                # an object was detected
                if len(good) > self.min_points:
                    self.object_counters[name] += 1
                    src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([scene_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if coordinates is None:
                        h, w, c = obj_img.shape
                    else:
                        _, _, w, h = coordinates

                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = np.int32(cv2.perspectiveTransform(pts, M))
                    # dst contains the coordinates of the vertices of the drawn rectangle

                    # get the min and max x and y coordinates of the object
                    x, y, w, h = self._extract_rectangle(dst)

                    # transform homography into a simpler data structure
                    dst = np.array([point[0] for point in dst], dtype=np.int32)

                    # check the object's height, width and area according to the parameters in the config file
                    if w < self.min_object_width or h < self.min_object_height or w * h < self.min_object_area:
                        break

                    # returns the homography and a rectangle containing o object
                    return dst, [x, y, w, h]

        return None, None

    def from_image(self, frame):
        self.current_frame += 1
        # Our operations on the frame come here

        scene_kp, scene_des = self.detector.detectAndCompute(frame, None)

        detected_objects = {}

        for obj_features in self.object_features:
            homography, rct = self._detect_object(obj_features,
                                                  [frame, scene_kp, scene_des])

            if rct is not None:
                ymin = rct[1]
                xmin = rct[0]
                ymax = rct[1] + rct[3]
                xmax = rct[0] + rct[2]
                object_name = obj_features[0]

                if object_name not in detected_objects:
                    detected_objects[object_name] = []

                detected_objects[object_name].append((ymin, xmin, ymax, xmax))

                text_point = (homography[0][0], homography[1][1])
                homography = homography.reshape((-1, 1, 2))
                cv2.polylines(frame, [homography], True, (0, 255, 255), 10)

                cv2.putText(frame, object_name + ': ' + str(self.object_counters[object_name]), text_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 0), 2)

        return frame, detected_objects

    @staticmethod
    def _extract_rectangle(dst):
        """
        Extract a rectangle from an OpenCV homography
        """
        min_y = min(dst[x, 0][1] for x in range(4))
        max_y = max(dst[x, 0][1] for x in range(4))
        min_x = min(dst[x, 0][0] for x in range(4))
        max_x = max(dst[x, 0][0] for x in range(4))
        return min_x, min_y, max_x - min_x, max_y - min_y
