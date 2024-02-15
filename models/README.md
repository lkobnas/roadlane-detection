# Pre-trained models

### High error rate
1. [model_canny_edge](model_canny_edge.h5) : Canny edge detection with Gaussian blur
2. [model_thresholding](model_thresholding.h5) : Thresholding coupled with Gaussian blur
3. [model_adaptive_thresholding](model_adaptive_thresholding.h5) : Adaptive thresholding complemented by Gaussian blur
4. [texmodel_hsl_filter_binary](model_hsl_filter_binary.h5) : HSL-filtered binary mask, combined with Gaussian blur

### Medium error rate
5. [model_hsl_filter_3_layer](model_hsl_filter_3_layer.h5) : HSL-filtered binary mask combined with the original image, along with Gaussian blur
6. [model_hsl_filter_1_layer](model_hsl_filter_1_layer.h5) : Greyscale image integrated with HSL-filtered binary mask and Gaussian blur

### Low error rate
7. [model_hsl_filter_1_layer_enhanced](model_hsl_filter_1_layer_enhanced.h5) : Greyscale image integrated with HSL-filtered binary mask, Gaussian blur, and reduced background weight.