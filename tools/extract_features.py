import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from argparse import ArgumentParser
import numpy as np
import torch

from datasetapi.voc_dataset.voc_data_processing import generate_xml_and_image_list
from datasetapi.voc_dataset.voc_data_processing import get_label_wh_xy_minmax

import mmcv
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from mmdet.datasets.pipelines import Compose
from mmdet.datasets import replace_ImageToTensor
from mmdet.apis import init_detector

def calculate_features(model, img, last_layer=1):
    """Calculate backbone feature(s) with the detector.
       the model and img are obtained based on mmdetection.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
        last_layer (int)：which layer of feature map will be extract and returned

    Returns:
        backbone_outputs: embeddings of the given layer of the backbone.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # prepare data
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)

    # build the data pipeline
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        # print(model.backbone)
        backbone_outputs = model.backbone(data['img'][0])
        # torch.save(backbone_outputs, "./features_tensor.pt")
        # for backbone_output in backbone_outputs:
        #    print(backbone_output.shape)
    return backbone_outputs[-last_layer]


def extract_features_array(config, checkpoint, device,
                           txt_path, xml_folder, image_folder,
                           # json_file,
                           last_layer, save_features):
    """
    Extract feature list of given dataset
    :param config: mmdetection config file
    :param checkpoint: mmdetection checkpoint file
    :param device: GPU number
    :param txt_path: VOC txt filepath
    :param xml_folder: VOC xml filepath
    :param image_folder: VOC image folder path
    :param json_file: COCO json filepath
    :param last_layer: which layer's embeddings will be output
    :param save_features: save the embeddings or not
    :return: feature_list: features of specified backbone layer of every image
             label_list: corresponding label of feature (ONLY can be used when each image contains only ONE object)
    """
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)
    # get test img list
    xml_file_list, image_list = generate_xml_and_image_list(txt_path, xml_folder, image_folder) #VOC format
    # image_list = get_coco_img_bbox(json_file) # COCO format

    # calculate backbone features
    feature_list = []
    label_list = []
    for img in image_list:
        print(img) #print img filename
        backbone_features = calculate_features(model, img, last_layer)
        # label_list ONLY can be used when each image only contains ONE box
        xml_file = xml_file_list[image_list.index(img)]
        label, _, _, _ = get_label_wh_xy_minmax(xml_file)
        label_list.append(label)
        feature_list.append(backbone_features.cpu().numpy())
    if save_features:
        np.save("saved_features_%s" %args.last_layer, np.array(feature_list))
        np.save("saved_features_labels_%s" % args.last_layer, np.array(label_list))
    print("shape of feature list is: ", np.array(feature_list).shape)
    return feature_list, label_list

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file of mmdet model')
    parser.add_argument('checkpoint', help='Checkpoint file of mmdet model')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--txt_path', '-t', default=None, help='path of the txt file for VOC datasetapi')
    parser.add_argument('--xml_folder', '-x', default=None, help='folder of the xml file for VOC datasetapi')
    parser.add_argument('--image_folder', '-i', default=None, help='folder of images')
    parser.add_argument('--last_layer', '-l', default=None, type=int, help='the last i th feature map')
    parser.add_argument('--save_features', '-s', default=False, type=bool, help='save the features')
    args = parser.parse_args()

    # VOC datasetapi format
    if args.txt_path and args.xml_folder and args.image_folder:
        extract_features_array(args.config, args.checkpoint, args.device, # basic info
                            args.txt_path, args.xml_folder, args.image_folder, # VOC params
                            args.last_layer, args.save_features) # output params
