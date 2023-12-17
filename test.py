import cv2
import json
import numpy as np
import osmlocator as ol

settings = ol.getDefaultSetting()
locator = ol.OsmLocator(settings)

img = cv2.imread('cli_tool/imgs/sml_00001.png', cv2.IMREAD_GRAYSCALE)
if img is None: exit()
binImg = (img<200).astype(np.uint8)*255
markLoctions = locator.locate(binImg)
print(markLoctions)

import osmlocator as ol

jcont = json.load(open('cli_tool/imgs/sml_00001.json'))
gtMarkLocs = jcont['marks_location'] 
#gtMarkLocs = [{'x':100,'y':100},{'x':200,'y':200}]
score = ol.evaluateXY(gtMarkLocs, markLoctions, img.shape)
print(score)
