# VGG16_PyTorch

This repository includes a VGG16 architecture-based python code with PyTorch. 

## VGG16

VGG16 is a deep convolutional neural network (CNN) architecture for image classification. It was originally developed by the Visual Geometry Group (VGG) at the University of Oxford and is one of the most widely used CNN architectures for image recognition tasks.
The architecture of VGG16 consists of the following layers:
### Input Layer: 
The input layer takes in an image tensor of size (224 x 224 x 3) for standard color images.
### Convolutional Layers: 
The VGG16 architecture contains 13 convolutional layers, each using small, 3x3 filters to detect local features in the input image. The number of filters in each layer increases gradually, from 64 in the first layer to 512 in the last convolutional layer. These layers use a rectified linear unit (ReLU) activation function to introduce non-linearity into the model.
### Pooling Layers: 
The VGG16 architecture also includes 5 max-pooling layers, which reduce the spatial dimensions of the feature maps by taking the maximum value of each region. This helps reduce the computational cost of the network and makes it more robust to changes in the position of objects in the image.
### F ully Connected Layers: 
After the convolutional and pooling layers, the feature maps are flattened and passed through two fully connected (FC) layers with 4096 neurons each, followed by a final FC layer with 1000 neurons for the final image classification.
### Softmax Activation: 
The final layer uses a softmax activation function to produce a probability distribution over the 1000 possible classes.
\\ Overall, the VGG16 architecture is a simple, yet effective model that has been widely used and tested on various image recognition tasks, achieving state-of-the-art results on many benchmark datasets.

## DATASET: 
[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
\
This is image data of Natural Scenes around the world.
\ 
This Data contains around 25k images of size 150x150 distributed under 6 categories.
\ 
{'buildings' -> 0,
\ 
'forest' -> 1,
\
'glacier' -> 2,
\ 
'mountain' -> 3,
\
'sea' -> 4,
\
'street' -> 5 }
\
The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction.
This data was initially published on https://datahack.analyticsvidhya.com by Intel to host a Image classification Challenge.
