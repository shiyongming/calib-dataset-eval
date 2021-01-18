import argparse
import xml.dom.minidom

def calculate_wh(xml_path=None):
    xml_tree = xml.dom.minidom.parse(xml_path)
    rootNode = xml_tree.documentElement
    objects = rootNode.getElementsByTagName("object")
    wh = []
    
    for obj in objects:
        # box = obj.getElementsByTagName("bndbox")
        class_name = str(obj.getElementsByTagName("name")[0].firstChild.data)
        xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
        ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
        xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
        ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
        w = xmax - xmin
        h = ymax - ymin
        wh.append([class_name,w,h])
    return wh

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--xml_path', '-x', default=None, help='path of the xml file')
    args = parser.parse_args()
    wh = calculate_wh(args.xml_path)
    
    