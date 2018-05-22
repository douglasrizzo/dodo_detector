# Object detection

This a Python package I made to make object detection easier. Besides the dependencies listed on `setup.py`, it also depends on the [OpenCV 3 nonfree/contrib packages](https://github.com/opencv/opencv_contrib) (which include the SURF and SIFT keypoint detection algorithms) and the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). The documentation on their repos teach everything you need to know in order to install it.

Since this package is not on PyPi, you can install it via `pip` like this:

    pip install git+https://github.com/douglasrizzo/dodo_detector.git

## How to use

The package has two types of detector, a keypoint-based detector and a Single Shot Detector.

### Keypoint-based detector

The keypoint-based object detector uses OpenCV 3 keypoint detection and description algorithms (namely, SURF, SIFT and [RootSIFT](https://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/)) together with feature matching algorithms in order to detect textures from a database directory on an image. I basically followed [this tutorial](https://docs.opencv.org/3.4.1/d1/de0/tutorial_py_feature_homography.html) and implemented it in a more organized way.

Example on running a keypoint-based detector:

    from dodo_detector import KeypointObjectDetector
    KeypointObjectDetector.KeypointObjectDetector('/path/to/my/database').from_camera(0)

The database directory must have the following structure:

    database_dir
        beer_can
            img1.jpg
            img2.jpg
            img3.jpg
        milk_box
            hauihu.jpg
            172812.jpg
            you_require_additional_pylons.jpg
        chocolate_milk
            .
            .
            .
        .
        .
        .

Basically, the top-level directory will contain subdirectory. The name of each subdirectory is the class name the program will return during detection. Inside each subdirectory is a collection of image files, whose keypoints will be extracted by the `KeypointObjectDetector` during the object construction. The keypoints will then be kept in-memory while the object exists.

You can then use the methods provided by the detector to detect objects in your images, videos or camera feed.

### Single-shot detector

This detector uses TensorFlow Object Detection API. Follow their tutorial on how to train your neural network. The training procedure will give you a _frozen inference graph_, which is a `.pb` file containing the network weights, as well as a _label map_, which is a text file with extension `.pbtxt` containing the names of your object classes.

When creating the SSDObjectDetector, the path to the frozen inference graph and label map must be passed, as well as the number of classes.

Example on running a single-shot detector:

    from dodo_detector import SSDObjectDetector
    SSDObjectDetector.SSDObjectDetector('path/to/frozen/graph.pb', 'path/to/labels.pbtxt', 5).from_camera(0)

Have fun!
