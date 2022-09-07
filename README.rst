Dodo's object detection package
===============================

This is a package that implements two types of object detection algorithms and provides them as Python classes, ready to be instantiated and used. The first algorithm uses a pipeline which consists of OpenCV keypoint detection and description algorithms, followed by feature matching and positioning using homography. Basically, `this tutorial <https://docs.opencv.org/3.4.1/d1/de0/tutorial_py_feature_homography.html>`__.

The second one uses any pre-trained convolutional network from the `TensorFlow Object Detection API <https://github.com/tensorflow/models/tree/master/research/object_detection>`__. Basically, `this tutorial <https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb>`__.

I've made an effort to keep the package compatible with both Python 2.7 + TensorFlow 1 and Python 3 + TensorFlow 2.

Why
---

After having to follow the same computer vision tutorials to implement object detection every time I needed it for a project, I decided I had enough and created this Python package with detection methods I often use, ready to be used out of the box. I hope this makes things easier not only for me, but for others.

Dependencies
------------

For the keypoint-based object detector:

- `OpenCV 3 nonfree/contrib packages <https://github.com/opencv/opencv_contrib>`__, which include the SURF [1]_ and SIFT [2]_ keypoint detection algorithms

For the TensorFlow-based object detectors:

- if you plan to use Python 2.7 and TensorFlow 1: `TensorFlow Object Detection API v1 <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md>`__
- if you plan to use Python 3 and TensorFlow 2: `TensorFlow Object Detection API v2 <https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md>`__

Follow their respective documentation pages to install them.

.. note::

    There is an ``opencv-contrib`` package available on PyPI, but I have never tried installing it instead of doing the aforementioned process.

Installation
------------

Since this package is not on PyPI, you can install it via ``pip`` like this:

.. code-block:: sh
    
    pip install git+https://github.com/douglasrizzo/dodo_detector.git

OpenCV is a hard dependency and is installed via the PyPI ``opencv-python`` package. If you already have OpenCV installed (*e.g.* from source), edit *setup.py* and remove the hard dependency before installing.

As for TensorFlow, either install the specific TensorFlow version that is going to be used, or create a virtualenv to run this package along with your TensorFlow version of choice. Make sure to also have the Object Detection API installed. Only the API compatible with your TensorFlow and Python versions should be on your ``PYTHONPATH``.

Usage
-----

The package has two types of detector, a keypoint-based detector and a detector that uses convolutional neural networks from the TensorFlow object detection API.

All detectors have a common interface, with three methods:

- ``from_camera`` takes a camera ID and uses OpenCV to read a frame stream, which is displayed on a separate window;
- ``from_video`` receives a video file and also displays the detection results on a window;
- ``from_image`` receives a single RGB image as a numpy array and returns a tuple containing an image with all the detected objects marked in it, and a dictionary containing object classes as keys and their detection information in tuples. Some classifiers return only bounding boxes, others return an additional confidence level. An example with one apple and two oranges detected in an image: ::

    {'person': [
        {'box': (204, 456, 377, 534), 'confidence': 0.9989906},
        {'box': (182, 283, 370, 383), 'confidence': 0.99848276},
        {'box': (181, 222, 368, 282), 'confidence': 0.9979938},
        {'box': (184, 37, 379, 109), 'confidence': 0.9938652},
        {'box': (169, 0, 371, 66), 'confidence': 0.98873794},
        {'box': (199, 397, 371, 440), 'confidence': 0.96926546},
        {'box': (197, 108, 365, 191), 'confidence': 0.96739936},
        {'box': (184, 363, 377, 414), 'confidence': 0.945458},
        {'box': (195, 144, 363, 195), 'confidence': 0.92953676}
    ]}

Keypoint-based detector
~~~~~~~~~~~~~~~~~~~~~~~

The keypoint-based object detector uses OpenCV 3 keypoint detection and description algorithms (namely, SURF [1]_, SIFT [2]_ and RootSIFT [3]_) to extract features from a database of images provided by the user. These features are then compared to features extracted from a target image, using feature matching algorithms also provided by OpenCV, to find the desired objects from the database in the target image.

Since OpenCV has no implementation of RootSIFT, I stole `this one <https://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/>`__.

Example on running a keypoint-based detector:

.. code-block:: python

    from dodo_detector.detection import KeypointObjectDetector
    detector = KeypointObjectDetector('/path/to/my/database_dir')
    marked_image, obj_dict = detector.from_image(im)

The database directory must have the following structure:

::

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

Basically, the top-level directory will contain subdirectories. The name of each subdirectory is the class name the program will return during detection. Inside each subdirectory is a collection of image files, whose keypoints will be extracted by the ``KeypointObjectDetector`` during the object construction. The keypoints will then be kept in-memory while the object exists.

You can then use the methods provided by the detector to detect objects in your images, videos or camera feed.

Convolutional neural network detectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These detectors use the TensorFlow Object Detection API. In order to use them, you must either train your own neural network using their API, or provide a trained network. I have a concise `tutorial <https://gist.github.com/douglasrizzo/c70e186678f126f1b9005ca83d8bd2ce>`__ on how to train a neural network for TensorFlow 2, with other useful links.

Python 2.7 or TensorFlow 1
**************************

The training procedure will give you the *frozen inference graph*, which is a ``.pb`` file; and a *label map*, which is a text file with extension ``.pbtxt`` containing the names of your object classes.

This type of detector must be pointed towards the paths for the frozen inference graph and label map. The number of classes is inferred from the contents of the label map.

Example on running the detector:

.. code-block:: python
    
    # load an image as a numpy array
    import numpy as np
    from PIL import Image
    im = np.array(Image.open('image.jpg'))

    # create the detector, pointing to the pre-trained model and the label map
    from dodo_detector.detection import TFObjectDetectorV1
    detector = TFObjectDetectorV1('path/to/frozen/graph.pb', 'path/to/labels.pbtxt')

    # use the detector to find objects in an image
    marked_image, objects = detector.from_image(im)
    # list objects found. locations are given in tuples in the format (ymin, xmin, ymax, xmax)
    objects
    
    {'person': [
        {'box': (204, 456, 377, 534), 'confidence': 0.9989906},
        {'box': (182, 283, 370, 383), 'confidence': 0.99848276},
        {'box': (181, 222, 368, 282), 'confidence': 0.9979938},
        {'box': (184, 37, 379, 109), 'confidence': 0.9938652},
        {'box': (169, 0, 371, 66), 'confidence': 0.98873794},
        {'box': (199, 397, 371, 440), 'confidence': 0.96926546},
        {'box': (197, 108, 365, 191), 'confidence': 0.96739936},
        {'box': (184, 363, 377, 414), 'confidence': 0.945458},
        {'box': (195, 144, 363, 195), 'confidence': 0.92953676}
    ]}

TensorFlow 2
************

After training and exporting a model, a directory called ``saved_model`` will be created, whose contents are used by *dodo_detector* to load the model into memory. Another file that is needed is the *label map*, which is a text file with extension ``.pbtxt`` containing the names of your object classes.

This type of detector must be pointed towards the paths of the ``saved_model`` directory and label map. The number of classes is inferred from the contents of the label map.

Example on running the detector:

.. code-block:: python

    # load an image as a numpy array
    import numpy as np
    from PIL import Image
    im = np.array(Image.open('image.jpg'))

    # create the detector, pointing to the pre-trained model and the label map
    from dodo_detector.detection import TFObjectDetectorV2
    detector = TFObjectDetectorV2('path/to/frozen/graph.pb', 'path/to/labels.pbtxt')

    # use the detector to find objects in an image
    marked_image, objects = detector.from_image(im)
    # list objects found. locations are given in tuples in the format (ymin, xmin, ymax, xmax)
    objects
    
    {'person': [
        {'box': (204, 456, 377, 534), 'confidence': 0.9989906},
        {'box': (182, 283, 370, 383), 'confidence': 0.99848276},
        {'box': (181, 222, 368, 282), 'confidence': 0.9979938},
        {'box': (184, 37, 379, 109), 'confidence': 0.9938652},
        {'box': (169, 0, 371, 66), 'confidence': 0.98873794},
        {'box': (199, 397, 371, 440), 'confidence': 0.96926546},
        {'box': (197, 108, 365, 191), 'confidence': 0.96739936},
        {'box': (184, 363, 377, 414), 'confidence': 0.945458},
        {'box': (195, 144, 363, 195), 'confidence': 0.92953676}
    ]}

Have fun!

.. rubric:: References

.. [1] H. Bay, A. Ess, T. Tuytelaars, and L. Van Gool, “Speeded-up robust features (SURF),” Computer vision and image understanding, vol. 110, no. 3, pp. 346–359, 2008.
.. [2] D. G. Lowe, “Object recognition from local scale-invariant features,” in Proceedings of the Seventh IEEE International Conference on Computer Vision, 1999, vol. 2, pp. 1150–1157.
.. [3] R. Arandjelović and A. Zisserman, “Three things everyone should know to improve object retrieval,” in 2012 IEEE Conference on Computer Vision and Pattern Recognition, 2012, pp. 2911–2918.
