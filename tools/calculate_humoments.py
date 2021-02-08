import argparse
import cv2 as cv

def calculate_humoments(image=None, image_path=None, isRGB=True, roi=None):
    """
    Get classes hu moments from given image
    :param image: image numpy array
    :param image_path: path of image file
    :param isRGB: RGB color image or not
    :param roi: all bonding boxes of given image [ [class_name, [xmin, ymin, xmax, ymax], ... ], ... [] ]
    :return: classes: bbox class list of given image,
             HuMoments_1: HU moments_1 of all bboxes in the given image
             HuMoments_2: HU moments_2 of all bboxes in the given image
    """

    if image_path is not None:
        image = cv.imread(image_path)
    elif image is not None:
        image = image
    classes = []
    HuMoments_1 = []
    HuMoments_2 = []
    for i, box in enumerate(roi):
        cls = box[0]
        xmin = box[1][0]
        ymin = box[1][1]
        xmax = box[1][2]
        ymax = box[1][3]
        cropped_image = image[ymin:ymax, xmin:xmax]
        if cropped_image.any():
            if isRGB:
                # cropped_image_b = cropped_image[:,:,0]
                # cropped_image_g = cropped_image[:,:,1]
                # cropped_image_r = cropped_image[:,:,2]
                # moments_b = cv.moments(cropped_image_b)
                # moments_g = cv.moments(cropped_image_g)
                # moments_r = cv.moments(cropped_image_r)
                # HuMoments_b = cv.HuMoments(moments_b)
                # HuMoments_g = cv.HuMoments(moments_g)
                # HuMoments_r = cv.HuMoments(moments_r)
                cropped_image_gray = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
                moments_gray = cv.moments(cropped_image_gray)
                HuMoments = cv.HuMoments(moments_gray)
            else:
                moments = cv.moments(cropped_image)
                HuMoments = cv.HuMoments(moments)
            classes.append(cls)
            HuMoments_1.append(HuMoments[0])
            HuMoments_2.append(HuMoments[1])
        else:
            continue
        # print('!!!!!!!!!!!!!!', classes)
    return classes, HuMoments_1, HuMoments_2


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--image', '-i', default=None, help='Numpy array of input image')
    parser.add_argument('--image_path', '-im', default=None, help='image path of the input image')
    parser.add_argument('--isRGB', '-is', default=True, help='is RGB color image')
    parser.add_argument('--roi', '-r', default=None, type=list, help='bonding box of object')
    args = parser.parse_args()
    HuMoments = calculate_humoments(args.image, args.image_path, args.isRGB, args.roi)
    # print(HuMoments)
    # cv.imshow("",image)
    # cv.waitKey(0)
    