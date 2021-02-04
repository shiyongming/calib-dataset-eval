from datasetapi.coco_dataset.cocoapi.coco import COCO

def get_coco_wh_xyminmax(annFile=None, image_root=None):

    coco = COCO(annFile)

    length = len(coco.anns)
    wh_list = []
    xyminmax_list = []
    ids = []
    images = []
    for i, annIdx in enumerate(coco.anns):
        ann = coco.loadAnns(annIdx)
        if len(ann) > 1:
            raise AssertionError('One or some annotations contain more than on element')
        category_id = ann[0]['category_id']
        ids.append(category_id)
        image_id = ann[0]['image_id']
        images.append(image_id)
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
    images_list = list(set(images))
    return ids_list, images, wh_list, xyminmax_list

def generate_calibset(annFile=None, percentage=10):
    coco = COCO(annFile)
    annIds = coco.anns.keys()
    length = len(annIds)

    ids = []
    wh_list = []
    xyminmax_list = []
    images = []
    calib_ids = []
    calib_wh_list = []
    calib_xyminmax_list = []
    calib_images = []

    for i, annIdx in enumerate(annIds):
        ann = coco.loadAnns(annIdx)

        if len(ann) > 1:
            raise AssertionError('One or some annotations contain more than on element')

        category_id = ann[0]['category_id']
        image_id = ann[0]['image_id']
        bbox = ann[0]['bbox']
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]

        if (i % 100) < percentage / 1:
            calib_ids.append(category_id)
            calib_images.append(image_id)
            calib_wh_list.append([category_id, w, h])
            calib_xyminmax_list.append([category_id, [x, y, x + w, y + h]])  # cat_id, xmin, ymin, xmax, ymax

        else:
            ids.append(category_id)
            images.append(image_id)
            wh_list.append([category_id, w, h])
            xyminmax_list.append([category_id, [x, y, x + w, y + h]])  # cat_id, xmin, ymin, xmax, ymax

        if (i % 1000 == 0):
            print('Finished %ik of %ik annotations' % (i / 1000, length / 1000))

    ids_list = list(set(ids))
    calib_ids_list = list(set(calib_ids))

    # images_list = list(set(images))
    # calib_images_list = list(set(calib_images))
    print('Use %i of %i annotations for calibration' % (len(calib_xyminmax_list), length))

    return ids_list, images, wh_list, xyminmax_list, \
           calib_ids_list, calib_images, calib_wh_list, calib_xyminmax_list

def get_coco_img_bbox(annFile=None):
    imgs = []
    bboxes = []
    coco = COCO(annFile)
    for img in (coco.imgs):
        imgs.append(img)
        annIdx = coco.getAnnIds(imgIds=[img])
        anns = coco.loadAnns(annIdx)
        temp_boxes = []
        for ann in anns:
            cls = ann['category_id']
            bbox = (int(ann['bbox'][0]), # xmin = x
                    int(ann['bbox'][1]), # ymin =
                    int(ann['bbox'][2] + ann['bbox'][0]), # xmax = x+w
                    int(ann['bbox'][3] + ann['bbox'][1])) # ymax = y+h
            temp_boxes.append([cls, bbox])
        bboxes.append(temp_boxes)
    return imgs, bboxes

