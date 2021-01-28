import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from argparse import ArgumentParser
import numpy as np
import torch

from dataset.voc_dataset import generate_xml_and_image_list
from dataset.voc_dataset import get_label_wh_xy_minmax

import mmcv
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter

from mmdet.datasets.pipelines import Compose
from mmdet.datasets import replace_ImageToTensor
from mmdet.apis import calculate_features, init_detector

def calculate_features(model, img, last_layer=1):
    """Calculate backbone feature(s) with the detector.
       the model and img are obtained based on mmdetection.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
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
                           last_layer, save_features):
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)
    # get test img list
    xml_file_list, image_list = generate_xml_and_image_list(txt_path, xml_folder, image_folder)
    # calculate backbone features
    feature_list = []
    label_list = []
    for img in image_list:
        print(img)
        xml_file = xml_file_list[image_list.index(img)]
        backbone_features = calculate_features(model, img, last_layer)
        label, _, _, _ = get_label_wh_xy_minmax(xml_file)
        label_list.append(label)
        #featuted_list.append(backbone_features)
        feature_list.append(backbone_features.cpu().numpy())
    if save_features:
        np.save("saved_features_%s" %args.last_layer, np.array(feature_list))
    print("shape of feature list is: ", np.array(feature_list).shape)
    return feature_list, label_list

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--txt_path', '-t', default=None, help='path of the txt file')
    parser.add_argument('--xml_folder', '-x', default=None, help='folder of the xml file')
    parser.add_argument('--image_folder', '-i', default=None, help='folder of images')
    parser.add_argument('--last_layer', '-l', default=None, type=int, help='the last i th feature map')
    parser.add_argument('--save_features', '-s', default=False, type=bool, help='save the features')
    args = parser.parse_args()
    extract_features_array(args.config, args.checkpoint, args.device,
                           args.txt_path, args.xml_folder, args.image_folder,
                           args.last_layer, args.save_features)
