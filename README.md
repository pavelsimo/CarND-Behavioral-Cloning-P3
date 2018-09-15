# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals/steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/001_original.jpg "Image center"
[image2]: ./examples/002_after_flip.jpg "Image after flip"
[image3]: ./examples/003_after_brightness_contrast.jpg "Brightness and contrast adjustments"
[image4]: ./examples/004_after_brightness_contrast.jpg "Brightness and contrast adjustments"
[image5]: ./examples/005_samples_distribution.png "Training set distribution"
[image6]: ./examples/006_model_accuracy.png "Model accuracy"
[image7]: ./examples/007_model_loss.png "Model loss"
[image8]: ./examples/008_after_crop.jpg "After crop"
[image9]: ./examples/010_img_left.jpg "Image left"
[image10]: ./examples/009_img_center.jpg "Image center"
[image11]: ./examples/011_img_right.jpg "Image right"


My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* video.mp4 a video recording of the vehicle driving autonomously

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

I ran into an issue where the ```drive.py``` exceed the memory available of my GPU, I fixed by controlling the memory
for a gpu process, this can be tweak using the tensor flow configuration option ```config.gpu_options.per_process_gpu_memory_fraction```. 

The ```model.py``` file contains the code for training and saving the convolution neural network. The file shows the 
pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

My model consist of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   					    |
| Cropping         		| output 65x320x3   					        |  
| Convolution 5x5     	| 2x2 stride, valid padding, output 31x158x25   |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, output 14x77x36    |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, output 5x37x48     |
| RELU					|												|
| Convolution 3x3	    | output 3x35x64                                |
| RELU					|												|
| Convolution 3x3	    | output 1x33x64                                |
| RELU					|												|
| Flatten	      	    | input 1x33x64, output 2112 				    |
| Fully connected		| input 2112, output 100       			    |
| Dropout				| Keep probability 0.5						    |
| Fully connected		| input 100, output 50       					|
| Dropout				| Keep probability 0.5						    |
| Fully connected		| input 50, output 10       					|
| Fully connected		| input 10, output 1       					    |

Some important characteristics of the model:

* The data is normalized using the Keras lambda function ```x / 127.5 - 0.5``` 
* The model includes RELU layers to introduce nonlinearity 
* The model contains two dropout layers in order to reduce overfitting
* The model used an adam optimizer, so the learning rate was not tuned manually

Model hyper-parameters:

* Epochs: 3
* Batch Size: 16

Using the udacity simulator, I collected a total of 109,900 images. Picture below shows the training set distribution:

![alt text][image5]

My data collection strategies were the following:

* Reckless center lane driving: I drove as fast as possible in the first track always aiming for the center lane,
when driving mistakes were made, I attempt to recover the best I could, thus allowing the model to learn recovery strategies.

* Forward and backward track driving: I drove the track forward and backwards, this help to reduce the bias caused by dominating 
left turns in the first track.

* Curve slow driving: This strategy consisted in driving slowly on the curves, this has the following benefits. First, 
facilitated the collection of curves samples, the majority of the track was straight paths, this improves the representation
of the curves within the training set. Secondly, improve the steering angle collection on the curves, since I drove slowly
was easy to take an optimal driving path. 

* Avoid left or right training set skew: By constantly looking at the training set distribution, I took decisions whether 
to drive the track forward or backward, this avoid the bias towards turning in a particular direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I took as starting point the network described in the NVIDIA paper End to End Learning for Self-Driving Cars. Since this 
model was designed specifically for the task at hand, I thought was an excellent starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and 
validation set. I found that my first model had a low mean squared error on the training set but a high mean 
squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added two Dropout layers with a keep probability of 0.5, this allow the model to generalized, thus not
overfitting the dataset.

After testing the model on track 1, there were a few spots where the vehicle fell off the track, to improve 
the driving behavior in these cases, I augmented the training set as follow:

* Brightness and contrast adjustments:

![alt text][image1] ![alt text][image3] ![alt text][image4]

* Horizontally flipping the images, and inverting the steering angle: 

![alt text][image1] ![alt text][image2]  

* Cropping the region of interest, so the classifier could focus on the relevant part of the image:

![alt text][image1] ![alt text][image8] 

* Making use of the both left and right images, and adding the appropriate steering correction angle:

![alt text][image9] ![alt text][image10] ![alt text][image11] 

As I mentioned before, after the collection process, I had 109,900 number of data points. after the augmenting step
I ended up with around 769,300 images, neat!

I finally randomly shuffled the data set and put 2% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under 
fitting. The ideal number of epochs was between around 3, after that amount the improvements of the classifier were neglectable. 
I was able to tell the classifier was neither overfitting nor underfitting the training set by observing the following 
model loss and accuracy graph:

![alt text][image7]  

![alt text][image6]


