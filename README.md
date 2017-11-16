#**Traffic Sign Recognition** 

##Writeup Template

---

**Aim of project is to build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/gray_newimg.jpg "Grayscaling"
[image4]: ./traffic-signs-images-web/ImageCropped1.png "Traffic Sign 1"
[image5]: ./traffic-signs-images-web/ImageCropped2.png "Traffic Sign 2"
[image6]: ./traffic-signs-images-web/ImageCropped3.png "Traffic Sign 3"
[image7]: ./traffic-signs-images-web/ImageCropped4.png "Traffic Sign 4"
[image8]: ./traffic-signs-images-web/ImageCropped5.png "Traffic Sign 5"
[image9]: ./examples/exampleImages.png "Traffic Sign Input"
[image10]: ./examples/Predictions.png "Traffic Sign Predictions"
[image11]: ./examples/outputFeatureMap.png "Output Feature Map"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

Here is a link to my [project code](https://github.com/sankalpdayal/carnd_term1_p2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Basic summary of the data set.

I used the numpy functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32) with 3 cahnnels RGB
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data for each class is distrubuted

![alt text][image1]

A sample of how the traffic sign image looks is 

![alt text][image4]

Following observations can be made about the data set.
Classes are very unevenly distributed. Hence it will be a good idea to create more images for the classes which have less representation.
Some of the images are dark. Hence it is a good idea to randomize brightess levels.

###Design and Test a Model Architecture

####1. Proprocessing of data 

As a first step, I decided to convert the images to grayscale because LeNet on grascale shows promising results and it reduces the amount of computation during training.

As a last step, I normalized the image data using the method (pixel - 128)/ 128 as now the values will range from -1 to 1.

As it was observed that distribution of each class was not same, I decided to generate additional data. 

To add more data to the the data set, I used the following techniques
1. Scaling
2. Translation
3. Warping
4. Changing Brightness
I chose these techniques as they mimic taking photo of the sign with different lighting and orientations.

Here is an example of a traffic sign image before and after grayscaling and an augmented image:

![alt text][image2]

Original training dataset size was 34799 which after adding more data became 51690. This gave increase of 1.4x

####2. Final Model

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 					|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 1x1x400 	|
| RELU					|												|
| Flatten				| Flatten layer 2,3 after max pool, output 800	|
| Droput				| prob 0.5 		   								|
| Fully connected		| Outputs 512    								|
| Droput				| prob 0.5 		   								|
| Fully connected		| Outputs 43    								|
| Softmax				| 		       									| 


####3. Training of model.

To train the model, I cost function as softmax cross entropy and regularization on l2 loss of each layer weights. The values of my hyperparameters are as follows
rate = 0.001
mu = 0
sigma = 0.1
reg_weight =1e-3

I trained for 50 epochs and used batch size of 128.

####4. Approach to reach to the final model

My final model results were:
* training set accuracy of 0.982
* validation set accuracy of 0.950
* test set accuracy of 0.927

If an iterative approach was chosen:
* I chose LeNet as the first model as it was easy to implement and already has shown promising results. 
* The problem with this architecture was it was limited in accuracy on validation test. My guess is that the base model was not elaborate enought to identify all the features in the traffic sign images.
* I adjusted the architecture by adding more convolutation layers with relu activation and max polling and fully connected layers. 
* Adding more layers started increasing accuracy on training data but still accuracy was less on validation data. Hence i increased regularziation to reduce overfitting.
* I added regularization as droput after fully connectded layer and l2 loss for all the weights of layers for the 
* To add more features in decision making I added features from previous layers as well before the fully connected layer. Also adde a droput after it to add regularization.
* I had tune the regularization weight to get the validation accuracy above 0.93

###Testing

####1. Random 5 images from interent for german traffic signs 

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The fifth image will be difficult to classify because I think that kind of image doesnt exist in the database.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![alt text][image10]

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.927

####3.Certainity on each prediction

The code for making predictions on my final model is located below the cell with title "Output Top 5 Softmax Probabilities For Each Image Found on the Web"

For the first 3 images, the model is relatively sure that this is a stop sign (probability of 0.9), and for others the confidence is low. First is also incorrect.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9         			| Children crossing 							| 
| .9     				| Speed Limit (30km/h) 							|
| .9					| General Caution								|
| .7	      			| Children crossing 			 				|
| .8				    | Speed limit (50 km/hr)   						|


### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Yisual output of your trained network's feature maps
I looked at the output of first conv layer to understand what features is the network looking at. Mostly it is trying to understand edges under different orientations. Here is the image showing the output.
![alt text][image10]