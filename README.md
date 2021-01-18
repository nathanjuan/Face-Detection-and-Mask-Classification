# Face Detection and Mask Classification
This project utilizes OpenCV Haar Cascades and a convolutional neural network classifier to identify any faces in a given image and determine whether each faces is masked or unmasked. The three main components of the project are the data loader, the face detection model using OpenCV Haar Cascades, and the mask classifier model using CNN classification.

&nbsp;
## [Data Loader](https://github.com/njuan123/Face-Detection-and-Mask-Classification/blob/main/data_loader.py)
The data loader is used to read images from a file location and create a dataset from them to be used to train the mask classification CNN. The dataset used to train the model in this project consisted of 1000 images of faces with masks on and 1000 images of unmasked faces. A masked face image and an unmasked face images that were a part of the dataset are displayed below.
Masked             |  Unmasked
:-------------------------:|:-------------------------:
![](https://github.com/njuan123/Face-Detection-and-Mask-Classification/blob/main/examples/dataset_masked.jpg)  |  ![](https://github.com/njuan123/Face-Detection-and-Mask-Classification/blob/main/examples/dataset_unmasked.jpg)

&nbsp;
## [Face Detector](https://github.com/njuan123/Face-Detection-and-Mask-Classification/blob/main/face_detector.py)
This part is used to detect faces within a given image. Faces are first identified using a frontal face cascade with low precision to ensure all possible faces are identified. Then, identified faces with overlap represent the same face and duplicates are removed, and finally eye cascades are filtered over the remaining faces to ensure that the original frontal faces are indeed faces. An example of the face detector is shown below.


#### Original Image
![](https://github.com/njuan123/Face-Detection-and-Mask-Classification/blob/main/examples/faces.jpg)

#### Detected Faces
![](https://github.com/njuan123/Face-Detection-and-Mask-Classification/blob/main/examples/detected_faces.jpg)

&nbsp;
## [Mask Classification](https://github.com/njuan123/Face-Detection-and-Mask-Classification/blob/main/mask_classifier_model.py)
The mask classifier is a convolutional neural network classification model that was trained on the 2000 image dataset from the data loader. After 20 epochs, the model attained a 99.5% accuracy with test images. This model was constructed from scratch and was not pre-trained. Classification examples are displayed below.

Masked     | Unmasked
:-------------------------:|:-------------------------:
<img src="https://github.com/njuan123/Face-Detection-and-Mask-Classification/blob/main/examples/masked.jpg" width=400 /> | <img src="https://github.com/njuan123/Face-Detection-and-Mask-Classification/blob/main/examples/unmasked.jpg" width=400 />

&nbsp;
## [Pipeline](https://github.com/njuan123/Face-Detection-and-Mask-Classification/blob/main/mask-classifier.ipynb)
Both the face detector and mask classification are brought together with the pipeline at the end of the IPython Notebook. For any inputted image, faces are detected, cropped, and classified, and the results are shown.

#### Example Input:
<img src="https://github.com/njuan123/Face-Detection-and-Mask-Classification/blob/main/examples/testimage.jpg" width=600 />

&nbsp;
#### Example Output:
![](https://github.com/njuan123/Face-Detection-and-Mask-Classification/blob/main/examples/modelresults.jpg)
