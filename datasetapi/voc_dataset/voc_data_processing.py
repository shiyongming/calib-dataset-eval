import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import argparse
import xml.dom.minidom

# from extract_info_from_voc import get_label_wh_xy_minmax
def get_label_wh_xy_minmax(xml_path=None):
    """
    Extract labels, wh, xy, bbox of VOC .xml file
    :param xml_path: .xml file path
    :return: labels: labels of every object in the given xml file, [ [cls_obj_1], ... ]
             wh: wh of bbox of every object in the given xml file, [ [cls_obj_1, w, h], ... ]
             xy: xy of bbox of every object in the given xml file, [ [cls_obj_1, x, y], ... ]
             xyminmax: bbox of every object in the given xml file, [ [cls_obj_1, [xmin, ymin, xmax, ymax]], ... ]
    """
    xml_tree = xml.dom.minidom.parse(xml_path)
    rootNode = xml_tree.documentElement
    objects = rootNode.getElementsByTagName("object")
    labels = []
    wh = []
    xy = []
    xyminmax =[]
    for obj in objects:
        # box = obj.getElementsByTagName("bndbox")
        class_name = str(obj.getElementsByTagName("name")[0].firstChild.data)
        xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
        ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
        xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
        ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
        w = xmax - xmin
        h = ymax - ymin
        x = (xmax + xmin)/2
        y = (ymax + ymin)/2
        # print(class_name)
        labels.append([class_name])
        wh.append([class_name,w,h]) # for wh visualization
        xy.append([class_name,x,y])
        xyminmax.append([class_name, [xmin, ymin, xmax, ymax]]) # used for cropping images and visualizing Hu moments
    return labels, wh, xy, xyminmax


def generate_xml_and_image_list(txt_path=None, xml_folder=None, image_root=None):
    """
    Generate image list from corresponding xml files.
    :param txt_path: VOC txt file path
    :param xml_folder: VOC xml folder path
    :param image_root: VOC image folder path, this is needed when the image filepath is relative path in xml files
    :return: xml_file_list: VOC xml filenames list,
             image_list: corresponding images list
    """
    f = open(txt_path)
    xml_file_list = []
    image_list = []
    for line in f:
        #generate xml list
        if str(line[-5]) == '.': # 'xxx.jpg'
            xml_filename = xml_folder + str(line[:-5]) + ".xml"
        else:
            xml_filename = xml_folder + str(line[:-1]) + ".xml"
        xml_file_list.append(xml_filename)
        
        #generate image filename list
        xml_tree = xml.dom.minidom.parse(xml_filename)
        rootNode = xml_tree.documentElement
        if image_root is not None:
            image_name = image_root + rootNode.getElementsByTagName("filename")[0].firstChild.data
        else:
            image_name = rootNode.getElementsByTagName("filename")[0].firstChild.data
        image_list.append(image_name)
    return xml_file_list, image_list

def generate_wh_xyminmax_list(txt_path=None, xml_folder=None):
    """
    Get bboxes lists of every image
    :param txt_path: VOC txt file path
    :param xml_folder: VOC xml folder path
    :return: wh_list: w & h of bboxes of every obj in each image.
                      [ [[cls_obj_1, w, h], ... [obj_m, w, h]]  # image_1
                      ...
                      [[cls_obj_1, w, h], ... [obj_x, w, h] ] # image_n
             xyminmax_list: bboxes of every obj in each image.
                            [ [[cls_obj_1, [xmin, ymin, xmax, ymax]], ... [obj_m, [xmin, ymin, xmax, ymax]]  # image_1
                            ...
                            [[cls_obj_1, [xmin, ymin, xmax, ymax]], ... [obj_x, [xmin, ymin, xmax, ymax]] ] # image_n
    """
    xml_list, _ = generate_xml_and_image_list(txt_path, xml_folder)
    wh_list = []
    xyminmax_list = []
    for xml_file in xml_list:
        _, wh_results, xy_results, xyminmax_results = get_label_wh_xy_minmax(xml_file)
        # for wh in wh_results:
        #     wh_list.append(wh)
        # for xyminmax in xyminmax_results:
        #     xyminmax_list.append(xyminmax)
        wh_list.append(wh_results) # For wh visualization
        xyminmax_list.append(xyminmax_results) # For Hu moments visualization

    return wh_list, xyminmax_list


def get_label_from_voc_xml(xml_list=None):
    for xml in xml_list:
        # generate xml list
        label, _, _, _ = get_label_wh_xy_minmax(xml)
    return label

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--txt_path', '-t', default=None, help='path of the txt file')
    parser.add_argument('--xml_folder', '-x', default=None, help='folder of the xml file')
    parser.add_argument('--image_root', '-i', default=None, help='prefix path for filename in xml file')
    args = parser.parse_args()
    xml_list, image_list = generate_xml_and_image_list(args.txt_path, args.xml_folder, args.image_root)
    wh_list, _ = generate_wh_xyminmax_list(args.txt_path, args.xml_folder)
    # labels = get_label_from_voc_xml(xml_list)
    # print(labels)