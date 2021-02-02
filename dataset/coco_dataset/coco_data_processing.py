from cocoapi.coco import COCO


annFile = r'C:\Users\yoshi\Documents\Codes\MyGithub\calib-dataset-eval\test_dataset\coco\instances_val2017.json'
coco = COCO(annFile)
length = len(coco.anns)
for i, annIdx in enumerate(coco.anns):
    ann = coco.loadAnns(annIdx)
    if len(ann)>1:
        raise AssertionError('One or some annotations contain more than on element')

    # if (i%100==0):
    #     print('finshed %i of %i' %(i, length))
print(ann[0].keys())
# annIdx = coco.getAnnIds(imgIds=[361621])
#
# catIdx = coco.getCatIds(catNms=[5])
# ann = coco.loadCats(catIdx)
# print((ann))
# print(ann[0].keys())
# print(ann[0]['bbox'])
# print(coco.getImgIds([361621]))
