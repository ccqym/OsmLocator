import os,cv2
import numpy as np
import util_extr as utile
import pandas.io.json as pjson
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('agg')

def binarizeThreshold(img, val=250, isShow=False):
    threshold = img<val
    threshold = threshold.astype(np.uint8)*255
    return threshold

def binarizeOtsu(img, bk=7, isShow=False):
    blurred = cv2.GaussianBlur(img, (bk, bk), 0)
    threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return threshold


def mkDirIfNotExist(path):
    path = path.replace('\\','/')
    cPath = path[0:path.rfind('/')]
    if not os.path.exists(cPath):
        print('mkdir or remkdir: %s' % cPath)
        os.makedirs(cPath)

#save to json file
def saveMarks(locs, fp):
    mkDirIfNotExist(fp)
    if fp is None: return
    data = {'locations':locs}
    with open(fp, "w" ) as writer:
        dumpS = pjson.dumps(data, indent=4, double_precision=3)
        writer.write(dumpS)
        print('saved:', fp)

#save to visulized output image.
def saveVisualization(pts, img, savePath, color, size, dpi=160, alpha=0.9):
    sm = ['o','o_','s','s_','D','D_','^','^_','v','v_','6','7','+','*']
    Xs = []
    Ys = []
    mks = {}
    for p in pts:
        if 'm' not in p.keys():
            k = 'o'
        else:
            k = sm[p['m']]
        if k not in mks.keys():
            mks[k] = {'Xs':[], 'Ys':[]}
        mks[k]['Xs'].append(p['x'])
        mks[k]['Ys'].append(p['y'])

    figSize = (img.shape[0]/72, img.shape[1]/72)
    fig = plt.figure(figsize=figSize)
    ax = plt.axes()
    ax.set_frame_on(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.imshow(img, cmap='gray')
    for k,v in mks.items():
        ax.scatter(v['Xs'], v['Ys'], s=size, c=color, marker='o', alpha=alpha)
    mkDirIfNotExist(savePath)
    fig.savefig(savePath, pad_inches=0, bbox_inches='tight', dpi=dpi)
    print('Located marks are saved in: ', savePath)
    plt.close()

def removeTextAndAxis(bimg, k=15, isShow=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    img_erosion = cv2.erode(bimg, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

    return img_dilation

