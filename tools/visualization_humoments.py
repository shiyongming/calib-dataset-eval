import argparse
import matplotlib.pyplot as plt
import numpy as np
from voc_data_processing import generate_wh_xyminmax_list, generate_xml_and_image_list
from kernels.calculate_humoments import calculate_humoments
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

def plot_humoments(train_image_list, train_xyminmax_list, calib_image_list, calib_xyminmax_list, class_idx):
    train_cls_name_list = []
    train_plot_hu_moments_1 = []
    train_plot_hu_moments_2 = []
    calib_cls_name_list = []
    calib_plot_hu_moments_1 = []
    calib_plot_hu_moments_2 = []
    
    for i in range(len(classes_names)):
        train_cls_name_list.append([])
        train_plot_hu_moments_1.append([])
        train_plot_hu_moments_2.append([])
        calib_cls_name_list.append([])
        calib_plot_hu_moments_1.append([])
        calib_plot_hu_moments_2.append([])
    
    for i, train_image_path in enumerate(train_image_list):
        cls_name, _, _, _, hu_moments = calculate_humoments(image_path=train_image_path, roi=train_xyminmax_list[i])
        train_cls_name_list[classes_names.index(cls_name)].append(cls_name)
        train_plot_hu_moments_1[classes_names.index(cls_name)].append(hu_moments[0])
        train_plot_hu_moments_2[classes_names.index(cls_name)].append(hu_moments[1])
    train_plot_list = [train_cls_name_list, train_plot_hu_moments_1, train_plot_hu_moments_2]

    for i, calib_image_path in enumerate(calib_image_list):
        cls_name, _, _, _, hu_moments = calculate_humoments(image_path=calib_image_path, roi=calib_xyminmax_list[i])
        calib_cls_name_list[classes_names.index(cls_name)].append(cls_name)
        calib_plot_hu_moments_1[classes_names.index(cls_name)].append(hu_moments[0])
        calib_plot_hu_moments_2[classes_names.index(cls_name)].append(hu_moments[1])
    calib_plot_list = [calib_cls_name_list, calib_plot_hu_moments_1, calib_plot_hu_moments_2]
    
    r = lambda: random.randint(0,255)
    colors=[]
    for i in range(len(classes_names)+1):
        color = ('#%02X%02X%02X' % (r(),r(),r()))
        colors.append(color)
    area = np.pi * 3**2

    if class_idx is None:
        for i in range(len(train_plot_list[0])):
            cls_name = train_plot_list[0][i][0]
            train_x = np.log(np.abs(train_plot_list[1][i])) # hu_moments_1
            train_y = np.log(np.abs(train_plot_list[2][i])) # hu_moments_1
            plt.scatter(train_x , train_y, s=area, c=colors[classes_names.index(cls_name)], alpha=0.1, label=cls_name)
            calib_cls_name = calib_plot_list[0][i][0]
            calib_x = np.log(np.abs(calib_plot_list[1][i])) # hu_moments_1
            calib_y = np.log(np.abs(calib_plot_list[2][i])) # hu_moments_2
            plt.scatter(calib_x ,calib_y , marker='x', c=colors[classes_names.index(calib_cls_name)], alpha=0.6)

    else:
        cls_name = train_plot_list[0][class_idx][0]
        train_x = np.log(np.abs(train_plot_list[1][class_idx])) # hu_moments_1
        train_y = np.log(np.abs(train_plot_list[2][class_idx])) # hu_moments_2
        plt.scatter(train_x ,train_y , s=area, c=colors[classes_names.index(cls_name)], alpha=0.1, label=cls_name)
        calib_cls_name = calib_plot_list[0][class_idx][0]
        calib_x = np.log(np.abs(calib_plot_list[1][class_idx])) # hu_moments_1
        calib_y = np.log(np.abs(calib_plot_list[2][class_idx])) # hu_moments_2
        plt.scatter(calib_x ,calib_y , marker='x', c=colors[classes_names.index(calib_cls_name)+1], alpha=0.5, label=calib_cls_name+"_calib")
        
        line_color = colors[classes_names.index(cls_name)+1]
        plt.plot([np.min(calib_x),np.max(calib_x)], [np.min(calib_y), np.min(calib_y)], c=line_color) #buttom
        plt.plot([np.min(calib_x),np.max(calib_x)], [np.max(calib_y), np.max(calib_y)], c=line_color) #top
        plt.plot([np.min(calib_x),np.min(calib_x)], [np.min(calib_y), np.max(calib_y)], c=line_color) #left
        plt.plot([np.max(calib_x),np.max(calib_x)], [np.min(calib_y), np.max(calib_y)], c=line_color) #right

    plt.xlabel('hu_moments_1')
    plt.ylabel('hu_moments_2')
    plt.legend()
    plt.savefig(r'hu_moments.png', dpi=600)
    plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train_txt_path', '-t', default=None, help='path of the training txt file')
    parser.add_argument('--calib_txt_path', '-c', default=None, help='path of the calibration txt file')
    parser.add_argument('--xml_folder', '-x', default=None, help='folder of the xml file')
    parser.add_argument('--class_idx', '-cl', default=None, type=int, help='index number of class')
    args = parser.parse_args()
   
    trian_xml_list, train_image_list = generate_xml_and_image_list(args.train_txt_path, args.xml_folder)
    _, train_xyminmax_list = generate_wh_xyminmax_list(args.train_txt_path, args.xml_folder)
   
    calib_xml_list, calib_image_list = generate_xml_and_image_list(args.calib_txt_path, args.xml_folder)
    _, calib_xyminmax_list = generate_wh_xyminmax_list(args.calib_txt_path, args.xml_folder)

    plot_humoments(train_image_list, train_xyminmax_list, calib_image_list, calib_xyminmax_list, args.class_idx)



    