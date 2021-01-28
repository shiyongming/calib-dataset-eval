# calib-dataset-eval
This repo aims to provide a tool to evaluate the calibration dataset before quantization.

# Major features
Calculate and visualize the features distribution of calibration dataset and training dataset.
Calculate and visualize the Hu moments distribution of calibration dataset and training dataset.
Calculate and visualize the labels distribution of calibration dataset and training dataset.


# Background
As we know, during the quantization process, a calibration dataset is necessary. The size of calibration dataset depends on tasks and models. In other words, different type of tasks or model structures need different samples amount for calibration.
There are two main reasons for the bad result of quantization. One is caused by the calibration dataset which doesn't cover the major distribution of training dataset. The other one is caused by model structure which is not suitable for quantization. (https://arxiv.org/pdf/1806.08342.pdf Quantizing deep convolutional networks for efficient inference: A whitepaper)
For the calibration dataset, let us take TensorRT as an example. For TensorRT, experiments indicate that about 500 images are sufficient for calibrating ImageNet classification networks. However, how many images for other tasks or networks? And, when someone obtained an unsatisfying quantization result, how to check whether it was caused by the calibration dataset reason or not?
This repo aims to provide a tool to evaluate the calibration dataset before quantization.
This tool evaluate the calibration dataset from three aspects: input images (Hu moments), output results (labels), and intermediate features (embeddings).

# Requirement
sklearn
opencv-python>=4.4.0.46
Pytorch1.6
https://github.com/open-mmlab/mmdetection or https://github.com/grimoire/mmdetection-to-tensorrt
