import argparse
import xml.dom.minidom

def get_label_wh_xy_minmax(xml_path=None):
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
        wh.append([class_name,w,h])
        xy.append([class_name,x,y])
        xyminmax.append([class_name, xmin, ymin, xmax, ymax])
    return labels, wh, xy, xyminmax

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--xml_path', '-x', default=None, help='path of the xml file')
    args = parser.parse_args()
    labels, wh, xy, xyminmax = get_label_wh_xy_minmax(args.xml_path)
    
    