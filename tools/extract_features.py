from argparse import ArgumentParser
from voc_data_processing import generate_xml_and_image_list
from mmdet.apis import calculate_features, init_detector
import numpy as np

def extract_features(config, checkpoint, device,
                     txt_path, xml_folder, image_folder,
                     last_layer, save_features):
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)
    # get test img list
    _, image_list = generate_xml_and_image_list(txt_path, xml_folder, image_folder)
    # calculate backbone features
    featuted_list = []
    for img in image_list:
        print(img)
        backbone_features = calculate_features(model, img, last_layer)
        #featuted_list.append(backbone_features)
        featuted_list.append(backbone_features.cpu().numpy())
    if save_features:
        np.save("saved_features_%s" %args.last_layer, np.array(featuted_list))
    print(np.array(featuted_list).shape)

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
    extract_features(args.config, args.checkpoint, args.device,
                     args.txt_path, args.xml_folder, args.image_folder,
                     args.last_layer, args.save_features)
