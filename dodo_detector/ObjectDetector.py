#!/usr/bin/env python
import cv2
from abc import ABCMeta, abstractmethod

from imutils.video import WebcamVideoStream


class ObjectDetector():
    __metaclass__ = ABCMeta

    @abstractmethod
    def from_image(self, frame):
        """
        Detects objects in an image

        :param frame: a numpy.ndarray containing the image where objects will be detected
        :return: a tuple containing the image, with objects marked by rectangles,
        and a dictionary listing objects and their locations as `(ymin, xmin, ymax, xmax)`
        """
        pass

    def _detect_from_stream(self, get_frame, stream, record=None, view=False):
        """
        This internal method detects objects from images retrieved from a stream, given a method that extracts frames from this stream

        :param get_frame: a method that extracts frames from the stream
        :param stream: an object representing a stream of images
        :param record: path to a video file to record the detection stream, or None for no recording
        :param view: whether to open an OpenCV window to view the results in real-time
        """
        ret, frame = get_frame(stream)

        if record is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(record, fourcc, 25.0, (640,480))

        while ret:
            marked_frame, objects = self.from_image(frame)

            if record is not None:
                out.write(marked_frame)

            if view:
                cv2.imshow("Detected Objects", marked_frame)
                if cv2.waitKey(1) == 27:
                    break  # ESC to quit

            ret, frame = get_frame(stream)

        cv2.destroyAllWindows()

        if record is not None:
            out.release()

    def from_camera(self, camera_id=0):
        """
        Detects objects in frames from a camera feed

        :param camera_id: the ID of the camera in the system
        """

        def get_frame(stream):
            frame = stream.read()
            ret = True
            return ret, frame

        stream = WebcamVideoStream(src=camera_id)

        stream.start()
        self._detect_from_stream(get_frame, stream)
        stream.stop()

    def from_video(self, filepath):
        """
        Detects objects in frames from a video file
        
        :param filepath: the path to the video file
        """

        def get_frame(stream):
            ret, frame = stream.read()
            return ret, frame

        stream = cv2.VideoCapture()
        stream.open(filename=filepath)

        self._detect_from_stream(get_frame, stream)
