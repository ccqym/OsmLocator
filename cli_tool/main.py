import os
import cv2
import time
import json
import parser
import numpy as np
import osmlocator as ol
import util_extr as utilE

if __name__ == '__main__':
    args = parser.getParser()

    #open the image file and convert to gray image
    print('Handling:', args.input)
    if args.is_hsv_s_channel:
        oriImg = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if len(oriImg.shape)==2:
            img = oriImg
        else:
            hsv = cv2.cvtColor(oriImg, cv2.COLOR_BGR2HSV)
            img = 255-hsv[:,:,1]
    else:
        img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        oriImg = img

    if img is None:
        print('failed to open image, maybe unsupportable file format.')
        exit()

    #remove text and axis in the input image
    if args.binarize_method=='ostu':
        bImg = utilE.binarizeOtsu(img, args.blur_kernel_size)
    elif args.binarize_method=='threshold':
        bImg = utilE.binarizeThreshold(img, args.binarize_threshold)
    if args.is_remove_text_and_axis:
        bImg = utilE.removeTextAndAxis(bImg, args.remove_kernel_size)

    #locate scatter marks
    startT = time.time()
    settings = parser.convertToSettings(args)
    locator = ol.OsmLocator(settings)
    markLoctions = locator.locate(bImg)
    endT = time.time()
    timeCons = endT - startT
    print('Costed:%d seconds.'%timeCons)

    #evaluate the results
    if args.gt is not None:
        jcont = json.load(open(args.gt, 'r'))
        gtMarkLocs = jcont['marks_location'] 
        perf = ol.evaluateXY(gtMarkLocs, markLoctions, bImg.shape)
        print('Performance:', perf)

    #save to json file
    if args.output is not None:
        utilE.saveMarks(markLoctions, args.output)

    #save to visulized output image.
    if args.visualize_outfile is not None:
        rgb = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
        utilE.saveVisualization(markLoctions, rgb, args.visualize_outfile, args.out_point_color, args.out_point_size)
