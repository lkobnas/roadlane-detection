# Self-Driving Sim: Enhancing Lane Detection and Steering Control
![](/media/banner-gif.gif)
## Background
Self-driving technology has emerged as a groundbreaking frontier in the automotive industry, revolutionising transportation with its potential to enhance road safety and efficiency. One crucial aspect of self-driving technology is **lane-keeping**, which enables vehicles to stay within designated road lanes while maintaining a safe and smooth trajectory. Lane-keeping technology utilises computer vision algorithms and machine learning techniques to process real-time visual data from the vehicle's cameras. By detecting lane markings and analysing road conditions, the self-driving system can precisely control the steering and ensure the vehicle remains within its intended lane.

## Prerequisite
1. [Udacity Driving Simulator](https://github.com/udacity/self-driving-car-sim)
2. Python / Jupyter Notebook

## Methods
![Methods](/media/Machine%20learning%20pipeline.png)

## Getting Started
1. [From the beginning](#data-collection)
2. [Model Deployment](#model-deployment)


### Data Collection
[Udacity’s self-driving simulator](https://github.com/udacity/self-driving-car-sim) is a powerful tool that allows users to gather driving data for training machine learning models. The simulator provides a virtual environment where users can navigate a car on various roadways, encountering diverse driving scenarios that are similar to real-world challenges. The driving simulator is equipped with two modes: Training mode and Autonomous mode.

In **Training mode**, driving data is collected by manually controlling the car while the program records user inputs and camera images at the same time. This data collection process forms the basis for training the machine learning model. 

In **Autonomous mode**, users can deploy their trained machine-learning models into the simulator, and the performance of the self-driving car is observed as it autonomously navigates through the simulated environment.

Once you've collected all the images in Training mode, please save the output data file with your code. Don't forget to modify the ```images_folder_name``` variable in the *drivingSimulator.ipynb* to match the name of the folder where your images are stored.

## All-in-one notebook
Run [*drivingSimulator.ipynb*](/src/drivingSimulator.ipynb) and follow the steps and instructions in the notebook.

This notebook contains the following steps:
1. Data Preparation
    - Data Balancing
    - Data Augmentation
    - Image Preprocessing

2. Model Training
    - Batch Generator
    - Nvidia’s CNN Model for Self-driving Car
    - Comma AI Steering Model

### More about Image Preprocessing
Image preprocessing is an essential part of improving the quality and suitability of the dataset for model training. It addresses a range of challenges, including noise reduction, handling different lighting conditions, and accommodating diverse road scenarios.

Run [*preprocessing_test.ipynb*](/src/preprocessing_test.ipynb) to explore the effectiveness and performance of various data preprocessing techniques. We will explore:
- Gaussian blur
- Canny edge detection
- Thresholding
- Adaptive thresholding
- Hue, Saturation and Lightness (HSL) filtering 

*How each method contributes to the accurate extraction of essential features in identifying road lanes?*

## Training
### Nvidia’s CNN Model for Self-driving Car
![nvidia-cnn-architecture](media/nvidia-cnn-architecture.png)

## Evaluation
### Model Deployment
If you'd like to test out the performance of a specific model in the simulator, you can use one of the pre-trained models available in the [models folder](/models). To do this, simply copy the model you're interested in testing and place it in the same directory as the [evaluate.py](evaluate.py) script. Don't forget to update the ```model_name``` variable in the script to match the name of the model you're using. This will allow you to evaluate the performance of the model directly in the simulator.

Then, run ```evaluate.py```

And run the simulator in **Autonomous Mode**, you can override the autonomous control by using the arrow keys. Navigate the vehicle to the left lane, then let it drive by itself!


## Challenges

## Approach

## References

[Udacity self-driving-car-sim](https://github.com/udacity/self-driving-car-sim)<br>
[End-to-End Deep Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)<br>
[A-Complete-Guide-to-Self-Driving-Car](https://www.codeproject.com/Articles/1273179/A-Complete-Guide-to-Self-Driving-Car)

