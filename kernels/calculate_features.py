import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
# from mmcv.runner import load_checkpoint

# from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.datasets import replace_ImageToTensor
from mmdet.models import build_detector


def calculate_features(model, img, last_layer):
    """Calculate backbone feature(s) with the detector.
       the model and img are obtained based on mmdetection.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
        last_layer: which layer of the feature map will be retured. Counting from back to front.

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
        print(model.backbone)
        backbone_outputs = model.backbone(data['img'][0])
        torch.save(backbone_outputs[-last_layer], "features_tensor.pt")
        for backbone_output in backbone_outputs:
            print(backbone_output.shape)
    return backbone_outputs[-last_layer]
