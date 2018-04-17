# Object detection

Example on running a keypoint-based detector:

    from dodo_detector import KeypointObjectDetector
    KeypointObjectDetector.KeypointObjectDetector('/path/to/my/database').from_camera(0)

Example on running a single-shot detector:

    from dodo_detector import SSDObjectDetector
    SSDObjectDetector.SSDObjectDetector('path/to/frozen/graph.pb', 'path/to/labels.pbtxt', 5).from_camera(0)

Have fun!
