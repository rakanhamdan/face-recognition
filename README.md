# Face Recognition with Deep Learning
## **Overview**
This notebook demonstrates a complete workflow for building a face recognition system using deep learning techniques. The system leverages transfer learning with the pre-trained EfficientNetB7 model, utilizing the CelebA dataset and TensorFlow for model training and evaluation. The model is trained on a GPU with mixed precision enabled to optimize performance. It employs a Siamese network architecture to learn facial similarities and uses Euclidean distance for face verification.

## **Workflow**

The primary task of this notebook is face recognition. It identifies whether two given images represent the same person.

## Environment Setup:

**GPU Configuration**: The notebook checks if a GPU is available and enables mixed precision if it is, which significantly improves training efficiency.
*Mixed Precision Training*:

Mixed precision uses both 16-bit and 32-bit floating-point types during training. This technique speeds up training and reduces memory usage while maintaining model accuracy.
Benefits: By using mixed precision, the model trains faster and more efficiently, especially when using a GPU.

*Kaggle API Key Configuration*:

The Kaggle API Key is configured to allow seamless downloading of datasets. This involves setting up the directory where Kaggle configuration files are stored.
*Dataset Handling:*

Download: The CelebA dataset is downloaded from Kaggle. It contains 202599 photos of different celebrities. You can found it [here](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

## **Data Preparation and Modeling**:

The training data is prepared in the form of pairs of images. Each pair is labeled with a 1 if the images are of the same person and a 0 if they are not. This setup is crucial for training the Siamese network.

### *Siamese Network*: 
The architecture involves a Siamese network, which consists of two identical subnetworks that share the same weights. This network learns to differentiate between similar and dissimilar pairs of images.
A Siamese network is a type of neural network architecture that learns to identify the similarity between two inputs by comparing their feature representations.
#### Functionality: 
During training, the network learns to produce similar feature vectors for images of the same person and different vectors for images of different people.
#### Transfer Learning:
The pre-trained DenseNet121 model is leveraged to improve the accuracy and efficiency of the face recognition system.
#### Euclidean Distance: 
The Euclidean distance between the feature vectors of the image pairs is computed to determine the similarity. A smaller distance indicates higher similarity, while a larger distance indicates dissimilarity.
### Evaluation and Inference:

The trained model is evaluated on a validation set to assess its accuracy and generalization capability.
Inference can be performed on new images to demonstrate the model's face recognition capabilities.
