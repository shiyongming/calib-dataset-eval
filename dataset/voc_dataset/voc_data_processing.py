import sys
sys.path.append('../../..\\')

import argparse
import xml.dom.minidom

from extract_info_from_voc import get_label_wh_xy_minmax

def generate_xml_and_image_list(txt_path=None, xml_folder=None):
    f = open(txt_path)
    xml_file_list = []
    image_list = []
    for line in f:
        #generate xml list
        xml_filename = xml_folder + str(line[:-4]) + "xml"
        xml_file_list.append(xml_filename)
        
        #generate image filename list
        xml_tree = xml.dom.minidom.parse(xml_filename)
        rootNode = xml_tree.documentElement
        image_name = rootNode.getElementsByTagName("filename")[0].firstChild.data
        image_list.append(image_name)
    return xml_file_list, image_list

def generate_wh_xyminmax_list(txt_path=None, xml_folder=None):
    xml_list, _ = generate_xml_and_image_list(txt_path, xml_folder)
    wh_list = []
    xyminmax_list = []
    for xml_file in xml_list:
        _, wh_results, xy_results, xyminmax_results = get_label_wh_xy_minmax(xml_file)
        for wh in wh_results:
            wh_list.append(wh)
        for xyminmax in xyminmax_results:
            xyminmax_list.append(xyminmax)

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
    args = parser.parse_args()
    xml_list, image_list = generate_xml_and_image_list(args.txt_path, args.xml_folder)
    wh_list, _ = generate_wh_xyminmax_list(args.txt_path, args.xml_folder)
    # labels = get_label_from_voc_xml(xml_list)
    # print(labels)