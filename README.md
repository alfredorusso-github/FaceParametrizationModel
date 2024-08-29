# FaceParametrizationModel
 
This is a convolutional neural network (CNN) developed for my thesis work at University of Milan. The purpose of this model is to predict the MakeHuman facial parameters from a simple photo. Basically, my CNN takes a 2D image of a face and estimates the 138 values that MakeHuman uses to build a detailed 3D model. 

To train the network, I created a lot of images and their corresponding parameters using a plugin I wrote specifically for MakeHuman. This plugin can be seen [clicking here](https://github.com/alfredorusso-github/Makehuman-FaceParametrization).

The result obtain are promising, anyway there are some thing to do in order to improve the model as well as the synthetica data generation in order to make it more accurate.