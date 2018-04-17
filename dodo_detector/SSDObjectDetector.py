#!/usr/bin/env python
from pprint import PrettyPrinter

import cv2
import numpy as np
import tensorflow as tf
from imutils.video import WebcamVideoStream
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from dodo_detector.ObjectDetector import ObjectDetector


class SSDObjectDetector(ObjectDetector):
    def __init__(self, path_to_frozen_graph, path_to_labels, num_classes):
        # load (frozen) tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Label maps map indices to category names, so that when our convolution
        # network predicts 5, we know that this corresponds to airplane.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)


    def from_image(self, frame):
        # object recognition begins here
        height, width, z = frame.shape
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                ################################
                # TensorFlow magic begins here #
                ################################
                image_np_expanded = np.expand_dims(frame, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name(
                    'image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = self.detection_graph.get_tensor_by_name(
                    'detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = self.detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name(
                    'num_detections:0')

                # Actual detection
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                scr = scores.tolist()
                box = boxes.tolist()
                clas = classes.tolist()

                detected_objects = {}

                # for each detected object
                for x in range(len(scr)):
                    # for each score the objecthas received for each class
                    for y in scr[x]:
                        # scores are given in descending order,
                        # so as soon as we find a low one, we stop looking
                        if y < 0.9:
                            break

                        clas_name = self.categories[int(clas[x][scr[x].index(y)]) - 1][
                            'name']  # nome do objeto dentro do dicionário de objetos
                        box_objects = box[x][
                            scr[x].index(y)
                        ]  # dados das caixas gráficas dos objetos normalizadas entre 0 e 1 ( box_objects = [Ymin, Xmin, Ymax, Xmax] )

                        ymin = int(box_objects[0] * height)
                        xmin = int(box_objects[1] * width)
                        ymax = int(box_objects[2] * height)
                        xmax = int(box_objects[3] * width)


                        if clas_name not in detected_objects:
                            detected_objects[clas_name] = []

                        detected_objects[clas_name].append((ymin,xmin,ymax,xmax))



                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                ################################

                return frame, detected_objects
