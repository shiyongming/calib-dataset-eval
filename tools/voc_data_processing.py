import os
import argparse
import xml.dom.minidom

def generate_xml_and_image_list(txt_path=None, xml_folder=None):
    f = open(txt_path)
    xml_file_list = []
    image_list = []
    for line in f:
        
        #generate xml list
        xml_filename = xml_folder + line[:-4] + "xml"
        xml_file_list.append(xml_filename)
        
        #generate image filename list
        xml_tree = xml.dom.minidom.parse(xml_filename)
        rootNode = xml_tree.documentElement
        image_name = rootNode.getElementsByTagName("filename")[0].firstChild.data
        image_list.append(image_name)
    return xml_file_list, image_list

# def generate_image_path_list(txt_path, xml_folder):
#     f = open(txt_path)
#     image_list = []
#     for line in f:
#         xml_filename = xml_folder + line[:-4] + "xml"
#         xml_tree = xml.dom.minidom.parse(xml_filename)
#         rootNode = xml_tree.documentElement
#         image_name = rootNode.getElementsByTagName("filename")[0].firstChild.data
#         image_list.append(image_name)
#     return image_list    


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--txt_path', '-t', default=None, help='path of the txt file')
    parser.add_argument('--xml_folder', '-x', default=None, help='folder of the xml file')
    args = parser.parse_args()
    xml_list, image_list = generate_xml_and_image_list(args.txt_path, args.xml_folder)

    # print(image_list)