import argparse
import cv2 as cv

def calculate_humoments(image=None, image_path=None, isRGB=True, roi=None):
    '''
    image: image numpy array
    roi: bonding box [class_name, xmin, ymin, xmax, ymax]
    '''
    # roi = ['abn1',35,317,111,384]
    cls_name = roi[0]
    xmin = int(roi[1])
    xmax = int(roi[3])
    ymin = int(roi[2])
    ymax = int(roi[4])
    if image is not None:
        cropped_image = image[ymin:ymax,xmin:xmax]
    else:
        image = cv.imread(image_path)
        cropped_image = image[ymin:ymax,xmin:xmax]
    
    if isRGB:
        cropped_image_b = cropped_image[:,:,0]
        cropped_image_g = cropped_image[:,:,1]
        cropped_image_r = cropped_image[:,:,2]
        cropped_image_gray = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)
        moments_b = cv.moments(cropped_image_b)
        moments_g = cv.moments(cropped_image_g)
        moments_r = cv.moments(cropped_image_r)
        moments_gray = cv.moments(cropped_image_gray)
        HuMoments_b = cv.HuMoments(moments_b)
        HuMoments_g = cv.HuMoments(moments_g)
        HuMoments_r = cv.HuMoments(moments_r)
        HuMoments_gray = cv.HuMoments(moments_gray)
        return [cls_name, HuMoments_b, HuMoments_g, HuMoments_r, HuMoments_gray]

    else:
        moments = cv.moments(cropped_image)
        HuMoments = cv.HuMoments(moments)
        return [cls_name, HuMoments]

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
    