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
Lighting is a crucial aspect that significantly impacts the performance of self-driving vehicles. To evaluate the efficacy of each preprocessing method, four images were selected from the dataset, enabling a visual representation of the effectiveness of each technique. These images covered a range of lighting conditions, including scenarios such as: <br>
(a) The road lane under full bright conditions; <br>
(b) An area shadowed by surrounding objects; <br>
(c) Shadows cast by trees onto the road;  <br>
(d) Shadows obscuring a section of the road.

![original](/media/4_original.png)

### 1. Canny Edge Detection

![canny_edge](/media/4_canny.png)

### 2. Thresholding

![thresholding](/media/4_thresholding.png)

### 3. Adaptive Thresholding

![adaptive_thresholding](/media/4_adaptive_thresholding.png)

### Observation
*Edge detection* algorithm used in this cases faced challenges due to rapidly changing lighting conditions on the track. We found that **shadows** in the images significantly affected the algorithm's performance, resulting in the absence of detectable edges, mistaken identification of shadows, and incorrect highlighting of road lane edges. The shadows pose a significant challenge to the algorithm, hindering the accurate identification of road lanes.

## Approach
Due to the limitations encountered with *edge segmentation* such as challenges in handling shadows, a shift in approach is necessary to explore alternative methods for effectively identifying road lanes under diverse conditions. One approach involves leveraging **colour segmentation techniques**. Unlike edge segmentation, which struggles to handle varying lighting conditions and complex scenarios, colour segmentation operates in the colour space and offers the potential to better isolate and identify the road lane based on its distinctive chromatic properties.

*Colour segmentation* is utilized in the **Hue, Saturation, Lightness (HSL)** colour space instead of the Red, Green, Blue (RGB) or Hue, Saturation, Value (HSV) colour spaces due to its advantages in addressing challenges and enhancing the accuracy of lane detection. The hue component represents the type of colour, saturation denotes the intensity of the colour, and lightness determines the brightness of the colour. The HSL colour space separates colour information from intensity instead of colour like RGB, making it particularly suitable for scenarios where lighting conditions vary significantly. By decoupling the luminance component from the chromatic components, the HSL colour space allows for more effective colour-based segmentation and facilitates better detection of lane boundaries under different lighting conditions.

![HSL-color-space](/media/HSL.png)

We can investigate and record the HSL values of the road lane with the [colour picker](/src/hsv_color_picker.py). By calculating the average HSL value of the road lane, we can construct two sets of condition equations, one for bright scenarios and another for shadowed conditions, which fulfil all the specified boundary conditions for extracting the road lane.
1. Condition for isolating white lane in the dark area
```
condition_1 = (h_channel > 100) & (h_channel < 120) & \
              ((s_channel - l_channel) > 20) & \
              ((s_channel - l_channel) < 40) & \
              (s_channel > 60)

```
2. Condition for isolating white lane in the bright area
```
condition_2 = (h_channel > 15) & (h_channel < 40) & \
              ((((s_channel - l_channel) > 175) & ((s_channel - l_channel) < 220)) |
              ((s_channel > 200) & (l_channel > 100)))

```

## Results

### 1. Binary mask created by applying HSL filter

![binary](/media/4_binary.png)

### 2. Combined image of the binary mask and the original image

![binary](/media/4_3layer.png)

### 3. Grey-scaled image of 2. (reduced noise)

![binary](/media/4_1layer.png)

### 4. Enhanced image with reduced weight in Hue and Saturation channel

![binary](/media/4_1layer_enhanced.png)

### Observation
The HSL filter was applied to isolate the road lane from the original images, both bright and dark, with successful results. However, deploying the masked image individually in real-world scenarios is not safe since it filters out all background information, potentially leading to dangerous situations if the real-time processing algorithm fails to capture the road lane. To address this issue, the binary image can be added back to the original image with an increased lightness value, enhancing the prominence of the white lane. Additionally, the weight of the hue and saturation channels can be reduced before the grey-scaling process, further optimizing the result. The enhanced image with reduced weight in Hue and Saturation channels has a dimmed background and less conspicuous shadows, effectively highlighting the road lane. This approach helps to diminish unwanted features in the images, increasing the likelihood of extracting essential features during the subsequent training process, ultimately improving the model's performance in lane detection and steering prediction.

## References

[Udacity self-driving-car-sim](https://github.com/udacity/self-driving-car-sim)<br>
[End-to-End Deep Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)<br>
[A-Complete-Guide-to-Self-Driving-Car](https://www.codeproject.com/Articles/1273179/A-Complete-Guide-to-Self-Driving-Car)<br>

