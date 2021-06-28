from PIL import Image, ImageDraw, ImageFont
import cv2
import sys, time
import numpy as np
import math


def make_oc(a, b):
    if (math.fabs(a) + math.fabs(b) >= 150) :
        return int( 10 * 15.0 / (2.0 * math.pi) * (math.atan2(a, b) + math.pi))
    else:
        return 15;

def ocm(tpimg, src):

    graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    sobelSrcX = cv2.Sobel(graySrc, cv2.CV_8UC1, 1, 0)
    sobelSrcY = cv2.Sobel(graySrc, cv2.CV_8UC1, 0, 1)    
    
    oc_height, oc_width = graySrc.shape
    tp_height, tp_width, tpchannels = tpimg.shape
    oc = np.zeros((oc_height, oc_width), np.uint8)

    # OC画像生成(ソース)高速版
    for x in range(oc_width):
        for y in range(oc_height):
            oc[y, x] = make_oc(sobelSrcY[y,x], sobelSrcX[y,x])

    # OC画像生成（テンプレート）
    grayTemp = cv2.cvtColor(tpimg, cv2.COLOR_BGR2GRAY)
    sobelSrcX = cv2.Sobel(grayTemp, cv2.CV_8UC1, 1, 0)
    sobelSrcY = cv2.Sobel(grayTemp, cv2.CV_8UC1, 0, 1)

    ocTp_height, ocTp_width = grayTemp.shape
    ocTemp = np.zeros((ocTp_height, ocTp_width), np.uint8)

    for x in range(ocTp_width):
        for y in range(ocTp_height):
            ocTemp[y, x] = make_oc(sobelSrcY[y,x], sobelSrcX[y,x])


    cost = 0
    score = oc_height * oc_width * 255
    x = 0
    y = 0
    #roi = 200
    x_start = 0
    y_start = 0
    x_end = oc_width - tp_width
    y_end = oc_height - tp_height
    step = 3

    for j in range(y_start, y_end, step):
        for i in  range(x_start, x_end, step):
            cost = 0
            
            roi = oc[j:j+tp_height, i:i+tp_width]
            cost = cv2.countNonZero(ocTemp - roi)

            if(score > cost):
                x = i
                y = j
                score = cost

    #2nd Step
    detail = 3
    xx = x
    yy = y

    if(xx>2 and yy>2 and xx < (oc_width-ocTp_width+2) and yy < (oc_height-ocTp_height+2)):
        for j in range(yy - detail, yy + detail + 1):
            for i in range(xx - detail, xx + detail + 1):
                if (i > oc_width - ocTp_width):
                    break
                if (j > oc_height - ocTp_height):
                    break
                cost = 0

                roi = oc[j:j+tp_height, i:i+tp_width]
                cost = cv2.countNonZero(ocTemp - roi)
                
                if (score > cost):
                    x = i
                    y = j
                    score = cost
    
    return x,y,score

if __name__ == '__main__':
    tpimg = cv2.imread("./template1.bmp")
    src = cv2.imread("./observed.bmp")

    x,y,score = ocm(tpimg, src)
    
    img = cv2.rectangle(src,(x,y),(x+ocTp_width,y+ocTp_height),(255,0,0),2)
    cv2.imwrite('result.png',img)

    print(x,y,score)