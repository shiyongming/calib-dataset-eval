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

## Requirement
- sklearn     
- opencv-python>=4.4.0.46     
- Pytorch1.6      
- https://github.com/open-mmlab/mmdetection or https://github.com/grimoire/mmdetection-to-tensorrt


## Getting start
git clone and build the docker
```shell
git clone https://github.com/shiyongming/calib-dataset-eval.git
cd calib-dataset-eval
docker build -t calib-dataset-eval:v0.1 docker/
```

calculate the Hu moments
```shell
python visualization/visualization_humoments.py \
    -t <path/of/train.txt>  # usually in VOC2007/ImageSets/Main/ \
    -c <path/of/calibration_dataset.txt>  # format like train.txt \
    -x <path/of/VOC2007/Annotations/>  # which contains .xml file \
    -i <prefix/path/for/filename/in/xml/file>  # (optional) prefix of the 'filename' item in .xml file \
    -cl <class index>  # which class you want to calculat and visualize \
```

calculate the Hu moments
```shell
python visualization/visualization_wh.py
    -t <path/of/train.txt>  # usually in VOC2007/ImageSets/Main/ 
    -c <path/of/calibration_dataset.txt>  # format like train.txt 
    -x <path/of/VOC2007/Annotations/>  # which contains .xml file 
    -cl <class index>  # which class you want to calculat and 
``` 

extract feature embeddings
```shell
python tools/extract_features.py 
    <path of mmdet config file> 
    <path of mmdet checkpoint file> 
    -t <path/of/VOC2007/.txt file> # which contains image list
    -x <path/of/VOC2007/Annotations/> # which contains .xml file
    -i <image root> # prefix of the 'filename' item in .xml file
    -l <layer number> # which layer of feature embedding you want to extract (count from back to front)
    -s <True or False> # save the extracted embeddings
```