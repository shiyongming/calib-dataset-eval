import argparse
import xml.dom.minidom

def get_wh_from_xml(xml_path=None):
    xml_tree = xml.dom.minidom.parse(xml_path)
    rootNode = xml_tree.documentElement
    objects = rootNode.getElementsByTagName("object")
    wh = []
    
    for obj in objects:
        # box = obj.getElementsByTagName("bndbox")
        xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
        ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
        xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
        ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
        w = xmax - xmin
        h = ymax - ymin
        wh.append([w,h])
    return wh

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--xml_path', '-x', default=None, help='path of the xml file')
    args = parser.parse_args()
    wh = get_wh_from_xml(args.xml_path)
    
    