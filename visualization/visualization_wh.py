import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import argparse
import matplotlib.pyplot as plt
import numpy as np
from dataset.voc_dataset.voc_data_processing import generate_wh_xyminmax_list
import random

random.seed(2021)

classes_names = ["abn1",
                 "abn2",
                 "abn3",
                 "abn4",
                 "abn5",
                 "abn6",
                 "abn7",
                 "abn8",
                 "abn9",
                 "abn10"]


def plot_wh(wh_list, calib_wh_list, class_idx):
    r = lambda: random.randint(0,255)
    colors=[]
    for i in range(len(classes_names)+1):
        color = ('#%02X%02X%02X' % (r(),r(),r()))
        colors.append(color)
    
    cls_name_list = []
    plot_w_list = []
    plot_h_list = []
    calib_cls_name_list = []
    calib_plot_w_list = []
    calib_plot_h_list = []
    for i in range(len(classes_names)):
        cls_name_list.append([])
        plot_w_list.append([])
        plot_h_list.append([])
        calib_cls_name_list.append([])
        calib_plot_w_list.append([])
        calib_plot_h_list.append([])

    for wh in wh_list:
        # cls_name = wh[0]
        # w = wh[1]
        # h = wh[2]
        cls_name_list[classes_names.index(wh[0])].append(wh[0])
        plot_w_list[classes_names.index(wh[0])].append(wh[1]) # w
        plot_h_list[classes_names.index(wh[0])].append(wh[2]) # h
    plot_list = [cls_name_list, plot_w_list, plot_h_list]
    
    for calib_wh in calib_wh_list:
        # calib_cls_name = calib_wh[0]
        calib_cls_name_list[classes_names.index(calib_wh[0])].append(calib_wh[0])
        calib_plot_w_list[classes_names.index(calib_wh[0])].append(calib_wh[1]) # w
        calib_plot_h_list[classes_names.index(calib_wh[0])].append(calib_wh[2]) # h
    calib_plot_list = [calib_cls_name_list, calib_plot_w_list, calib_plot_h_list]   
    
    area = np.pi * 3**2
    if class_idx is None:
        for i in range(len(plot_list[0])):
            cls_name = plot_list[0][i][0]
            x = plot_list[1][i] # w
            y = plot_list[2][i] # h
            plt.scatter(x ,y , s=area, c=colors[classes_names.index(cls_name)], alpha=0.1, label=cls_name)
            calib_cls_name = calib_plot_list[0][i][0]
            calib_x = calib_plot_list[1][i] # w
            calib_y = calib_plot_list[2][i] # h
            plt.scatter(calib_x ,calib_y , marker='x', c=colors[classes_names.index(calib_cls_name)], alpha=0.5)

    else:
        cls_name = plot_list[0][class_idx][0]
        x = plot_list[1][class_idx] # w
        y = plot_list[2][class_idx] # h
        plt.scatter(x ,y , s=area, c=colors[classes_names.index(cls_name)], alpha=0.1, label=cls_name)
        calib_cls_name = calib_plot_list[0][class_idx][0]
        calib_x = calib_plot_list[1][class_idx] # w
        calib_y = calib_plot_list[2][class_idx] # h
        plt.scatter(calib_x ,calib_y , marker='x', c=colors[classes_names.index(calib_cls_name)+1], alpha=0.6, label=calib_cls_name+"_calib")

        line_color = colors[classes_names.index(cls_name)+1]
        plt.plot([np.min(calib_x),np.max(calib_x)], [np.min(calib_y), np.min(calib_y)], c=line_color) #buttom
        plt.plot([np.min(calib_x),np.max(calib_x)], [np.max(calib_y), np.max(calib_y)], c=line_color) #top
        plt.plot([np.min(calib_x),np.min(calib_x)], [np.min(calib_y), np.max(calib_y)], c=line_color) #left
        plt.plot([np.max(calib_x),np.max(calib_x)], [np.min(calib_y), np.max(calib_y)], c=line_color) #right

    plt.xlabel('w')
    plt.ylabel('h')
    plt.legend()
    plt.savefig(r'wh.png', dpi=600)
    plt.show()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train_txt_path', '-t', default=None, help='path of the training txt file for VOC format')
    parser.add_argument('--calib_txt_path', '-c', default=None, help='path of the calibration txt file for VOC format')
    parser.add_argument('--xml_folder', '-x', default=None, help='folder of the xml file for VOC format')
    parser.add_argument('--class_idx', '-cl', default=None, type=int, help='index number of class')
    args = parser.parse_args()

    # VOC dataset format
    wh_list, _= generate_wh_xyminmax_list(args.train_txt_path, args.xml_folder)
    calib_wh_list, _ = generate_wh_xyminmax_list(args.calib_txt_path, args.xml_folder)
    plot_wh(wh_list, calib_wh_list, args.class_idx)
    
    # TODO COCO dataset
