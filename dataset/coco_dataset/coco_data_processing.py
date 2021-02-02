from dataset.coco_dataset.cocoapi.coco import COCO
def get_coco_wh_xyminmax(annFile=None):
    coco = COCO(annFile)
    # testImgId = 293794
    # coco.download(tarDir=r'test_dataset/coco', imgIds=[293794])
    # annIdx = coco.getAnnIds(imgIds=[testImgId])
    # anns = coco.loadAnns(annIdx)
    # image = cv.imread('test_dataset/coco/000000293794.jpg')
    # cv.imshow('',image)
    # cv.waitKey(0)
    # for ann in anns:
    #     bbox = (ann['bbox'])
    #     x = int(bbox[0])
    #     y = int(bbox[1])
    #     w = int(bbox[2])
    #     h = int(bbox[3])
    #     # print (bbox)
    #     ptLeftTop = (x,y)
    #     ptRightBottom = (x+w,y+h)
    #     print(ptRightBottom)
    #     cv.rectangle(image,ptLeftTop,ptRightBottom,(0,0,255),1,4)

    # cv.imshow('',image)
    # cv.waitKey(0)
    length = len(coco.anns)
    wh_list = []
    xyminmax_list = []
    ids = []
    for i, annIdx in enumerate(coco.anns):
        ann = coco.loadAnns(annIdx)
        if len(ann) > 1:
            raise AssertionError('One or some annotations contain more than on element')
        category_id = ann[0]['category_id']
        ids.append(category_id)
        bbox = ann[0]['bbox']
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        wh_list.append([category_id, w, h])
        xyminmax_list.append([category_id, x, y, x + w, y + h])  # cat_id, xmin, ymin, xmax, ymax
        if (i % 1000 == 0):
            print('Finished %ik of %ik annotations' % (i / 1000, length / 1000))
    ids_list = list(set(ids))

    return ids_list, wh_list, xyminmax_list
