import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import argparse
import matplotlib.pyplot as plt
import numpy as np
from dataset.voc_dataset.voc_data_processing import generate_wh_xyminmax_list, generate_xml_and_image_list
from dataset.coco_dataset.coco_data_processing import get_coco_wh_xyminmax, generate_calibset, get_coco_img_bbox
from tools.calculate_humoments import calculate_humoments
import random
random.seed(2021)

def plot_humoments(train_image_list, train_xyminmax_list,
                   calib_image_list, calib_xyminmax_list,
                   train_image_root=None, calib_image_root=None,
                   ids_or_names=None, plot_idx=None):
    train_cls_list = []
    train_plot_hu_moments_1 = []
    train_plot_hu_moments_2 = []
    calib_cls_list = []
    calib_plot_hu_moments_1 = []
    calib_plot_hu_moments_2 = []
    
    for i in range(len(ids_or_names)):
        train_cls_list.append([])
        train_plot_hu_moments_1.append([])
        train_plot_hu_moments_2.append([])
        calib_cls_list.append([])
        calib_plot_hu_moments_1.append([])
        calib_plot_hu_moments_2.append([])

    total = len(train_image_list)
    for i, train_image_name in enumerate(train_image_list):
        train_image_path = train_image_root + str(train_image_name).zfill(12) + '.jpg'
        cls_list, hu_moments_1, hu_moments_2 = calculate_humoments(image_path=train_image_path, roi=train_xyminmax_list[i])
        # if cls != -1:
        for j, cls in enumerate(cls_list):
            train_cls_list[ids_or_names.index(cls)].append(cls)
            train_plot_hu_moments_1[ids_or_names.index(cls)].append(hu_moments_1[j])
            train_plot_hu_moments_2[ids_or_names.index(cls)].append(hu_moments_2[j])
        # else:
        #     continue
        if (i % 1000 == 0):
            print('Finished %ik of %ik images' % (i / 1000, total / 1000))
    train_plot_list = [train_cls_list, train_plot_hu_moments_1, train_plot_hu_moments_2]

    total = len(calib_image_list)
    for i, calib_image_name in enumerate(calib_image_list):
        calib_image_path = calib_image_root + str(calib_image_name).zfill(12) + '.jpg'
        cls_list, hu_moments_1, hu_moments_2 = calculate_humoments(image_path=calib_image_path, roi=calib_xyminmax_list[i])
        for j, cls in enumerate(cls_list):
            calib_cls_list[ids_or_names.index(cls)].append(cls)
            calib_plot_hu_moments_1[ids_or_names.index(cls)].append(hu_moments_1[j])
            calib_plot_hu_moments_2[ids_or_names.index(cls)].append(hu_moments_2[j])
        # else:
        #     continue
        if (i % 1000 == 0):
            print('Finished %ik of %ik images' % (i / 1000, total / 1000))
    calib_plot_list = [calib_cls_list, calib_plot_hu_moments_1, calib_plot_hu_moments_2]
    
    r = lambda: random.randint(0,255)
    colors=[]
    for i in range(len(ids_or_names)+1):
        color = ('#%02X%02X%02X' % (r(),r(),r()))
        colors.append(color)
    area = np.pi * 2**2

    if plot_idx is None:
        for i in range(len(train_plot_list[0])):
            cls = train_plot_list[0][i][0]
            train_x = np.log(np.abs(train_plot_list[1][i])) # hu_moments_1
            train_y = np.log(np.abs(train_plot_list[2][i])) # hu_moments_1
            plt.scatter(train_x , train_y, s=area, c=colors[ids_or_names.index(cls)], alpha=0.1, label=cls)
            calib_cls = calib_plot_list[0][i][0]
            calib_x = np.log(np.abs(calib_plot_list[1][i])) # hu_moments_1
            calib_y = np.log(np.abs(calib_plot_list[2][i])) # hu_moments_2
            plt.scatter(calib_x ,calib_y , s=area, marker='x', c=colors[ids_or_names.index(calib_cls)], alpha=0.3)

    else:
        cls = train_plot_list[0][plot_idx][0]
        train_x = np.log(np.abs(train_plot_list[1][plot_idx])) # hu_moments_1
        train_y = np.log(np.abs(train_plot_list[2][plot_idx])) # hu_moments_2
        calib_cls = calib_plot_list[0][plot_idx][0]
        calib_x = np.log(np.abs(calib_plot_list[1][plot_idx])) # hu_moments_1
        calib_y = np.log(np.abs(calib_plot_list[2][plot_idx])) # hu_moments_2
        if type(cls) is str:
            plt.scatter(train_x ,train_y , s=area, c=colors[ids_or_names.index(cls)], alpha=0.1, label='trainset_cls:'+cls)
            plt.scatter(calib_x ,calib_y , s=area, marker='x', c=colors[ids_or_names.index(calib_cls)+1], alpha=0.3, label='calib_cls:'+cls)
        elif type(cls) is int:
            plt.scatter(train_x ,train_y , s=area, c=colors[ids_or_names.index(cls)], alpha=0.1, label='trainset_cls:'+cls)
            plt.scatter(calib_x ,calib_y , s=area, marker='x', c=colors[ids_or_names.index(calib_cls)+1], alpha=0.3, label='calib_cls:'+str(calib_cls))

        line_color = colors[ids_or_names.index(cls)+1]
        plt.plot([np.min(calib_x),np.max(calib_x)], [np.min(calib_y), np.min(calib_y)], c=line_color) #buttom
        plt.plot([np.min(calib_x),np.max(calib_x)], [np.max(calib_y), np.max(calib_y)], c=line_color) #top
        plt.plot([np.min(calib_x),np.min(calib_x)], [np.min(calib_y), np.max(calib_y)], c=line_color) #left
        plt.plot([np.max(calib_x),np.max(calib_x)], [np.min(calib_y), np.max(calib_y)], c=line_color) #right

    plt.xlabel('hu_moments_1')
    plt.ylabel('hu_moments_2')
    plt.legend()
    plt.savefig(r'hu_moments_coco2017.png', dpi=600)
    plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train_txt_path', '-t', default=None, help='path of the training txt file for VOC format')
    parser.add_argument('--calib_txt_path', '-c', default=None, help='path of the calibration txt file for VOC format')
    parser.add_argument('--xml_folder', '-x', default=None, help='folder of the xml file for VOC format')
    parser.add_argument('--train_json_path', '-tj', default=None, help='folder of the json file for COCO format')
    parser.add_argument('--calib_json_path', '-cj', default=None, help='folder of the json file for COCO format')
    parser.add_argument('--calib_percentage', '-cp', default=None, type=int, help='split percentage from calib_json to do calib')
    parser.add_argument('--train_image_root', '-ti', default=None, help='prefix path for filename in xml file')
    parser.add_argument('--calib_image_root', '-ci', default=None, help='prefix path for filename in xml file')
    parser.add_argument('--plot_cls_idx', '-p', default=None, type=int, help='index number of class')
    args = parser.parse_args()
    
    # VOC dataset format
    if (args.train_txt_path is not None) and (args.calib_txt_path is not None) and (args.xml_folder is not None) and \
            (args.train_json_path is None) and (args.calib_json_path is None):
        trian_xml_list, train_image_list = generate_xml_and_image_list(args.train_txt_path,
                                                                       args.xml_folder,
                                                                       args.image_root)
        _, train_xyminmax_list = generate_wh_xyminmax_list(args.train_txt_path, args.xml_folder)

        calib_xml_list, calib_image_list = generate_xml_and_image_list(args.calib_txt_path,
                                                                       args.xml_folder,
                                                                       args.image_root)
        _, calib_xyminmax_list = generate_wh_xyminmax_list(args.calib_txt_path, args.xml_folder)
        train_ids_list = ["abn1", "abn2", "abn3", "abn4", "abn5",
                         "abn6", "abn7", "abn8", "abn9", "abn10"]

    # COCO dataset format
    elif (args.train_txt_path is None) and (args.calib_txt_path is None) and (args.xml_folder is None) and \
            ((args.train_json_path is not None) or (args.calib_json_path is not None)):
        if args.calib_percentage is None:
            train_image_list, train_xyminmax_list = get_coco_img_bbox(args.train_json_path)
            calib_image_list, calib_xyminmax_list = get_coco_img_bbox(args.calib_json_path)


            # train_ids_list, _, _, _ = get_coco_wh_xyminmax(args.train_json_path)
            calib_ids_list, _, _, _ = get_coco_wh_xyminmax(args.calib_json_path)

    
    plot_humoments(train_image_list, train_xyminmax_list,
                   calib_image_list, calib_xyminmax_list,
                   args.train_image_root, args.calib_image_root,
                   calib_ids_list, args.plot_cls_idx)



    