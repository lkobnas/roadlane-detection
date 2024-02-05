import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
from matplotlib import pyplot as plt

sio = socketio.Server()

app = Flask(__name__)  # '__main__'
speed_limit = 10


def gaussian_blur_1(img):
    img = img[60:135, 10:, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


def thresholding(img):
    img = img[60:135, :, :]  # Crop upper part
    # Resize the image to a consistent size
    img = cv2.resize(img, (200, 66))
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    # Apply adaptive thresholding to enhance lane visibility
    _, thresholded = cv2.threshold(equalized, 150, 255, cv2.THRESH_BINARY)
    # # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(thresholded, (3, 3), 0)
    # expand image dimension
    expand = np.expand_dims(blurred, axis=2)
    # Normalize pixel values to [0, 1]
    normalized = expand / 255.0
    return normalized


def hsl_filter_1_layer(image):
    image = image[60:135, :, :]  # Crop upper part
    # Resize the image to a consistent size
    image = cv2.resize(image, (200, 66))

    # Convert the image to the HSL color space
    hsl_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # Split the HSL image into H, L, and S channels
    h_channel, s_channel, l_channel = cv2.split(hsl_image)

    # Conditions
    # condition_1 = (100 < h_channel < 120) & (55 < s_channel < 80) & (25 < l_channel < 45)  Dark
    condition_1 = (100<h_channel)&(h_channel< 120) & (65<s_channel)&(s_channel<80) & (25<l_channel)&(l_channel<55)
    # condition_2 = (20 < h_channel < 40) & (s_channel > 200) | (l_channel > 200)   Bright
    condition_2 = (20<h_channel)&(h_channel<40) & ((s_channel-l_channel)>175) & ((s_channel-l_channel)<220) #| (l_channel > 220)

    # Create a binary mask by combining the two conditions using logical OR
    combined_mask = np.logical_or(condition_1, condition_2).astype(np.uint8) * 255

    # Convert the binary mask to 3 channels for merging with the original image
    binary_image = cv2.merge((combined_mask, combined_mask, combined_mask))

    # Apply bitwise_and to keep only the white lane lines
    white_lane_image = cv2.bitwise_or(image, binary_image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(white_lane_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # expand image dimension
    expand = np.expand_dims(blurred, axis=2)

    # Normalize pixel values to [0, 1]
    normalized = expand / 255.0

    return normalized


def hsl_filter_1_layer_enhanced(img):
    image = img[60:135, :, :]  # Crop upper part
    # Resize the image to a consistent size
    image = cv2.resize(image, (200, 66))

    # Convert the image to the HSL color space
    hsl_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # Split the HSL image into H, L, and S channels
    h_channel, s_channel, l_channel = cv2.split(hsl_image)

    # Conditions
    # condition_1 = (100 < h_channel < 120) & (55 < s_channel < 80) & (25 < l_channel < 45)  Dark
    condition_1 = (90<h_channel)&(h_channel< 120) & ((s_channel-l_channel)>20) & ((s_channel-l_channel)<40) & (s_channel > 60)#(55<s_channel)&(s_channel<80) & (25<l_channel)&(l_channel<55)
    # condition_2 = (20 < h_channel < 40) & (s_channel > 200) | (l_channel > 200)   Bright
    condition_2 = (15<h_channel)&(h_channel<40) & ((((s_channel-l_channel)>175) & ((s_channel-l_channel)<220)) | ((s_channel>200)&(l_channel>100)))#| (l_channel > 220)

    # Create a binary mask by combining the two conditions using logical OR
    combined_mask = np.logical_or(condition_1, condition_2).astype(np.uint8) * 255

    # Convert the binary mask to 3 channels for merging with the original image
    binary_image = cv2.merge((combined_mask, combined_mask, combined_mask))

    # Reduce the saturation and lightness of original image
    s_channel = s_channel * 0.8
    l_channel = l_channel * 0.6
    reduced_image = cv2.merge((h_channel.astype(np.uint8), s_channel.astype(np.uint8), l_channel.astype(np.uint8)))

    # Apply bitwise_and to keep only the white lane lines
    white_lane_image = cv2.bitwise_or(reduced_image, binary_image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(white_lane_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # expand image dimension
    expand = np.expand_dims(blurred, axis=2)

    # Normalize pixel values to [0, 1]
    normalized = expand / 255.0

    return normalized


def hsl_filter_3_layer(image):
    image = image[60:135, :, :]  # Crop upper part
    # Resize the image to a consistent size
    image = cv2.resize(image, (200, 66))

    # Convert the image to the HSL color space
    hsl_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # Split the HSL image into H, L, and S channels
    h_channel, s_channel, l_channel = cv2.split(hsl_image)

    # Conditions
    # condition_1 = (100 < h_channel < 120) & (55 < s_channel < 80) & (25 < l_channel < 45)  Dark
    condition_1 = (90<h_channel)&(h_channel< 120) & ((s_channel-l_channel)>20) & ((s_channel-l_channel)<40) & (s_channel > 60)#(55<s_channel)&(s_channel<80) & (25<l_channel)&(l_channel<55)
    # condition_2 = (20 < h_channel < 40) & (s_channel > 200) | (l_channel > 200)   Bright
    condition_2 = (15<h_channel)&(h_channel<40) & ((((s_channel-l_channel)>175) & ((s_channel-l_channel)<220)) | ((s_channel>200)&(l_channel>100)))#| (l_channel > 220)

    # Create a binary mask by combining the two conditions using logical OR
    combined_mask = np.logical_or(condition_1, condition_2).astype(np.uint8) * 255

    # Convert the binary mask to 3 channels for merging with the original image
    binary_image = cv2.merge((combined_mask, combined_mask, combined_mask))

    # Reduce the saturation and lightness of original image
    s_channel = s_channel * 0.8
    l_channel = l_channel * 0.6

    reduced_image = cv2.merge((h_channel.astype(np.uint8), s_channel.astype(np.uint8), l_channel.astype(np.uint8)))

    # Apply bitwise_and to keep only the white lane lines
    white_lane_image = cv2.bitwise_or(reduced_image, binary_image)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(white_lane_image, (5, 5), 0)

    # Normalize pixel values to [0, 1]
    normalized = blurred / 255.0

    return normalized


def hsl_filter_binary(image):
    image = image[60:135, :, :]  # Crop upper part
    # Resize the image to a consistent size
    image = cv2.resize(image, (200, 66))

    # Convert the image to the HSL color space
    hsl_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # Split the HSL image into H, L, and S channels
    h_channel, s_channel, l_channel = cv2.split(hsl_image)

    # Conditions
    # condition_1 = (100 < h_channel < 120) & (55 < s_channel < 80) & (25 < l_channel < 45)  Dark
    condition_1 = (90<h_channel)&(h_channel< 120) & ((s_channel-l_channel)>20) & ((s_channel-l_channel)<40) & (s_channel > 60)#(55<s_channel)&(s_channel<80) & (25<l_channel)&(l_channel<55)
    # condition_2 = (20 < h_channel < 40) & (s_channel > 200) | (l_channel > 200)   Bright
    condition_2 = (15<h_channel)&(h_channel<40) & ((((s_channel-l_channel)>175) & ((s_channel-l_channel)<220)) | ((s_channel>200)&(l_channel>100)))#| (l_channel > 220)

    # Create a binary mask by combining the two conditions using logical OR
    combined_mask = np.logical_or(condition_1, condition_2).astype(np.uint8) * 255

    # Convert the binary mask to 3 channels for merging with the original image
    binary_image = cv2.merge((combined_mask, combined_mask, combined_mask))

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(binary_image, (5, 5), 0)

    # Normalize pixel values to [0, 1]
    normalized = blurred / 255.0

    return normalized


@sio.on('telemetry')
def telemetry(sid, data):
    if data is not None:
        speed = float(data['speed'])
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = preprocess_selection(image_processing_name, image)

        # Display the processed image in the window
        cv2.imshow('Real-time Display', image)
        # cv2.imshow("Real-time visualisation",cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        image = np.array([image])
        steering_angle = float(model.predict(image))
        print(steering_angle)
    else:
        speed = 0
        steering_angle = 0

    if (steering_angle > 0.5) | (steering_angle < -0.5):
        throttle = 1.0 - speed / 5
    else:
        throttle = 1.0 - speed / speed_limit

    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)
    cv2.waitKey(1)

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)


@sio.on('disconnect')
def disconnect(sid, environ):
    print('Disconnected')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


def preprocess_selection(function_name, img):
    # Check if the function exists
    if function_name in globals() and callable(globals()[function_name]):
        func = globals()[function_name]
        return func(img)
    else:
        print(f"Function '{function_name}' not found or not callable.")


def model_selection(name):
    if name == "model_hsl_filter_1_layer":
        processing_name = "hsl_filter_1_layer"
        return load_model('model_hsl_filter_1_layer.h5'), processing_name
    elif name == "model_hsl_filter_1_layer_enhanced":
        processing_name = "hsl_filter_1_layer_enhanced"
        return load_model('model_hsl_filter_1_layer_enhanced.h5'), processing_name
    elif name == "model_hsl_filter_1_layer_enhanced_commaai":
        processing_name = "hsl_filter_1_layer_enhanced"
        return load_model('model_hsl_filter_1_layer_enhanced_commaai.h5'), processing_name
    elif name == "model_hsl_filter_3_layer":
        processing_name = "hsl_filter_3_layer"
        return load_model('model_hsl_filter_3_layer.h5'), processing_name
    elif name == "model_hsl_filter_binary":
        processing_name = "hsl_filter_binary"
        return load_model('model_hsl_filter_binary.h5'), processing_name
    elif name == "model_gaussian_blur":
        processing_name = "gaussian_blur_1"
        return load_model('model_gaussian_blur.h5'), processing_name
    elif name == "model_thresholding":
        processing_name = "thresholding"
        return load_model('model_thresholding.h5'), processing_name
    elif name == "b":
        processing_name = ""
        return load_model('.h5'), processing_name
    else:
        print("Model not found")


if __name__ == '__main__':
    model_name = "model_hsl_filter_3_layer"
    cv2.namedWindow("Real-time Display", cv2.WINDOW_NORMAL)
    model, image_processing_name = model_selection(model_name)
    print(model.summary())

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)