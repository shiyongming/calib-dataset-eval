# calib-dataset-eval
This repo aims to provide a tool to evaluate the calibration dataset before quantization.

# Major features
Calculate and visualize the features distribution of calibration dataset and training dataset.

Calculate and visualize the Hu moments distribution of calibration dataset and training dataset.

Calculate and visualize the labels distribution of calibration dataset and training dataset.


# Background
As we know, during the quantization process, a calibration dataset is necessary. The size of calibration dataset depends on tasks and models. In other words, different type of tasks or model structures need different samples amount for calibration.

There are two main reasons for the bad result of quantization. One is caused by the calibration dataset which doesn't cover the major distribution of training dataset. The other one is caused by model structure which is not suitable for quantization. [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/pdf/1806.08342.pdf)

For the calibration dataset, let us take TensorRT as an example. For TensorRT, experiments indicate that about 500 images are sufficient for calibrating ImageNet classification networks. However, how many images for other tasks or networks? And, when someone obtained an unsatisfying quantization result, how to check whether it was caused by the calibration dataset reason or not?

This repo aims to provide a tool to evaluate the calibration dataset before quantization.
This tool evaluate the calibration dataset from three aspects (or levels): 
From the input level (input image), we calculate and campare the Hu moments distribution between training set and calibration set. 
For the output level (output result), we compare label distribution of each bonding box between training set and calibration set. 
For the intermediate level (intermediate feature), we calaulate and compare the intermediate (embeddings) between training set and calibration set.

## Requirement
- sklearn     
- opencv-python>=4.4.0.46     
- Pytorch1.7
- mmcv>=1.2.4 <1.3 
- https://github.com/open-mmlab/mmdetection or https://github.com/grimoire/mmdetection-to-tensorrt


## Getting start
git clone and build the docker
```shell
git clone https://github.com/shiyongming/calib-dataset-eval.git
cd calib-dataset-eval
docker build -t calib-dataset-eval:v0.1 docker/
```

Evaluate the Hu moments distribution
```python
python visualization/visualization_humoments.py 
    -t <path/of/train.txt>  # usually in VOC2007/ImageSets/Main/ 
    -c <path/of/calibration_dataset.txt>  # format like train.txt
    -x <path/of/VOC2007/Annotations/>  # which contains .xml file
    -i <prefix/path/for/filename/in/xml/file>  # (optional) prefix of the 'filename' item in .xml file
    -cl <class index>  # which class you want to calculat and visualize
```
![Hu moments ditribution](visualization/visualization_results/hu_moments.png) 


Evaluate the weight and height (label) distribution
```python
python visualization/visualization_wh.py
    -t <path/of/train.txt>  # usually in VOC2007/ImageSets/Main/ 
    -c <path/of/calibration_dataset.txt>  # format like train.txt 
    -x <path/of/VOC2007/Annotations/>  # which contains .xml file 
    -cl <class index>  # which class you want to calculate and visualize
``` 
![Labels ditribution](visualization/visualization_results/wh.png) 

Extract feature embeddings
```python
python tools/extract_features.py 
    <path of mmdet config file> 
    <path of mmdet checkpoint file> 
    -t <path/of/VOC2007/.txt file> # which contains image list
    -x <path/of/VOC2007/Annotations/> # which contains .xml file
    -i <image root> # prefix of the 'filename' item in .xml file
    -l <layer number> # which layer of feature embedding you want to extract (count from back to front)
    -s <boolean> # save the extracted embeddings
```

Evaluate the intermediate feature distribution
```python
python visualization/visualization_features.py
    -tf <numpy file which contains trainset embeddings>,
    -tn <numpy array which contains trainset embeddings>,
    -tl <numpy file which contains trainset labels>, # -tl can be set when each image only contains one label
    -cf <numpy file which contains calibset embeddings>,
    -cn <numpy array which contains calibset embeddings>,
    -cl <numpy file which contains calibset labels>, # -cl can be set when each image only contains one label
    -d <visualization dimension>
```
![embeddings distribution](visualization/visualization_results/features_distribution.png)