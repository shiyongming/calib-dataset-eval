import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import argparse
import matplotlib.pyplot as plt
import numpy as np
from datasetapi.voc_dataset.voc_data_processing import generate_wh_xyminmax_list
from datasetapi.coco_dataset.coco_data_processing import get_coco_wh_xyminmax, generate_calibset
import random

random.seed(2021)

# ids_or_names = ["abn1",
#                "abn2",
#                "abn3",
#                "abn4",
#                "abn5",
#                "abn6",
#                "abn7",
#                "abn8",
#                "abn9",
#                "abn10"]


def plot_wh(wh_list, calib_wh_list, ids_or_names=None, plot_idx=None):
    """

    :param wh_list: w & h list of training set
    :param calib_wh_list: w & h list of calibration set
    :param ids_or_names: category or id list
    :param plot_idx: which category or id will be plot
    :return:
    """
    r = lambda: random.randint(0,255)
    colors=[]
    for i in range(len(ids_or_names) + 1):
        color = ('#%02X%02X%02X' % (r(),r(),r()))
        colors.append(color)
    
    cls_list = []
    plot_w_list = []
    plot_h_list = []
    calib_cls_list = []
    calib_plot_w_list = []
    calib_plot_h_list = []

    for i in range(len(ids_or_names)):
        cls_list.append([])
        plot_w_list.append([])
        plot_h_list.append([])
        calib_cls_list.append([])
        calib_plot_w_list.append([])
        calib_plot_h_list.append([])
    for wh in wh_list:
        cls_list[ids_or_names.index(wh[0])].append(wh[0]) # cls_name_or_id = wh[0]
        plot_w_list[ids_or_names.index(wh[0])].append(wh[1])  # w = wh[1]
        plot_h_list[ids_or_names.index(wh[0])].append(wh[2])  # h = wh[2]
    plot_list = [cls_list, plot_w_list, plot_h_list]

    for calib_wh in calib_wh_list:
        # calib_cls_name = calib_wh[0]
        calib_cls_list[ids_or_names.index(calib_wh[0])].append(calib_wh[0])
        calib_plot_w_list[ids_or_names.index(calib_wh[0])].append(calib_wh[1])  # w
        calib_plot_h_list[ids_or_names.index(calib_wh[0])].append(calib_wh[2])  # h
    calib_plot_list = [calib_cls_list, calib_plot_w_list, calib_plot_h_list]

    area = np.pi * 2**2
    if plot_idx is None:
        for i in range(len(plot_list[0])):
            cls = plot_list[0][i][0]
            x = plot_list[1][i] # w
            y = plot_list[2][i] # h
            plt.scatter(x, y, s=area, c=colors[ids_or_names.index(cls)], alpha=0.1, label=cls)
            calib_cls = calib_plot_list[0][i][0]
            calib_x = calib_plot_list[1][i] # w
            calib_y = calib_plot_list[2][i] # h
            plt.scatter(calib_x, calib_y, s=area, marker='x', c=colors[ids_or_names.index(calib_cls)], alpha=0.5)

    else:
        cls = plot_list[0][plot_idx][0]
        x = plot_list[1][plot_idx] # w
        y = plot_list[2][plot_idx] # h
        calib_cls = calib_plot_list[0][plot_idx][0]
        calib_x = calib_plot_list[1][plot_idx]  # w
        calib_y = calib_plot_list[2][plot_idx]  # h
        if type(cls) is str:
            plt.scatter(x, y, s=area, c=colors[ids_or_names.index(cls)], alpha=0.1, label='trainset_cls:'+cls)
            plt.scatter(calib_x, calib_y, marker='x', c=colors[ids_or_names.index(calib_cls) + 1], alpha=0.3, label='calib_cls:'+calib_cls)

        elif type(cls) is int:
            plt.scatter(x, y, s=area, c=colors[ids_or_names.index(cls)], alpha=0.1, label='trainset_cls:'+str(cls))
            plt.scatter(calib_x, calib_y, s=area, marker='x', c=colors[ids_or_names.index(calib_cls) + 1], alpha=0.3, label='calib_cls:'+str(calib_cls))

        line_color = colors[ids_or_names.index(calib_cls) + 1]

        plt.plot([np.min(calib_x),np.max(calib_x)], [np.min(calib_y), np.min(calib_y)], c=line_color) #buttom
        plt.plot([np.min(calib_x),np.max(calib_x)], [np.max(calib_y), np.max(calib_y)], c=line_color) #top
        plt.plot([np.min(calib_x),np.min(calib_x)], [np.min(calib_y), np.max(calib_y)], c=line_color) #left
        plt.plot([np.max(calib_x),np.max(calib_x)], [np.min(calib_y), np.max(calib_y)], c=line_color) #right

    plt.xlabel('w')
    plt.ylabel('h')
    plt.legend()
    plt.savefig(r'wh distribution_coco2017.png', dpi=600)
    plt.show()
    # print(ids_or_names)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train_txt_path', '-tt', default=None, help='path of the training txt file for VOC format')
    parser.add_argument('--calib_txt_path', '-ct', default=None, help='path of the calibration txt file for VOC format')
    parser.add_argument('--xml_folder', '-x', default=None, help='folder of the xml file for VOC format')
    parser.add_argument('--train_json_path', '-tj', default=None, help='folder of the json file for COCO format')
    parser.add_argument('--calib_json_path', '-cj', default=None, help='folder of the json file for COCO format')
    parser.add_argument('--calib_percentage', '-cp', default=None, type=int, help='split percentage of calib_json for calib')
    parser.add_argument('--plot_cls_idx', '-p', default=None, type=int, help='index number of class')
    args = parser.parse_args()

    # VOC datasetapi format
    if (args.train_txt_path is not None) and (args.calib_txt_path is not None) and \
            (args.xml_folder is not None) and \
            (args.train_json_path is None) and (args.calib_json_path is None):
        wh_list, _= generate_wh_xyminmax_list(args.train_txt_path, args.xml_folder)
        calib_wh_list, _ = generate_wh_xyminmax_list(args.calib_txt_path, args.xml_folder)
        cls_list = list(set(wh_list[0]))
        calib_cls_list = list(set(calib_wh_list[0]))

    # COCO datasetapi format
    elif (args.train_txt_path is None) and (args.calib_txt_path is None) and \
            (args.xml_folder is None) and \
            ((args.train_json_path is not None) or (args.calib_json_path is not None)):
        # annFile = r'C:\Users\yoshi\Documents\Codes\MyGithub\calib-datasetapi-eval\test_dataset\coco\instances_val2017.json'
        if args.calib_percentage is None:
            cls_list, _, wh_list, _ = get_coco_wh_xyminmax(args.train_json_path)
            calib_cls_list, _, calib_wh_list, _ = get_coco_wh_xyminmax(args.calib_json_path)
        else:
            cls_list, _, wh_list, _ = get_coco_wh_xyminmax(args.train_json_path)
            _, _, _, _, calib_cls_list, _, calib_wh_list, _ = \
                generate_calibset(annFile=args.calib_json_path,
                                  percentage=args.calib_percentage)

    print('there are %i train categories' %len(cls_list))
    print('there are %i calib categories' %len(calib_cls_list))
    if len(calib_cls_list)< len(cls_list):
        raise AssertionError('please increase calib set')
    print('there are %i train bboxes' %len(wh_list))
    print('there are %i calib bboxes' %len(calib_wh_list))

    plot_wh(wh_list, calib_wh_list, cls_list, args.plot_cls_idx)
