import sys
import os
import argparse

from voc_data_processing import generate_xml_and_image_list
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from kernels.calculate_voc_wh import calculate_wh

def get_wh_list(txt_path=None, xml_folder=None):
    xml_list, _ = generate_xml_and_image_list(txt_path, xml_folder)
    wh_list = []
    for xml_file in xml_list:
        wh_results = calculate_wh(xml_file)
        for wh in wh_results:
            wh_list.append(wh)

    return wh_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--txt_path', '-t', default=None, help='path of the txt file')
    parser.add_argument('--xml_folder', '-x', default=None, help='folder of the xml file')
    args = parser.parse_args()
    wh_list = get_wh_list(args.txt_path, args.xml_folder)
    print((wh_list))


