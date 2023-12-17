import cv2
import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans

markCache = {}

def getConnCompAndSingleMarkSize(img, zoom=3):
    assert(img is not None)
    num, labels = cv2.connectedComponents(img)
    labelsDict = {}
    sizeD = {}
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            if not img[y,x]: continue
            k = labels[y][x]
            if k in labelsDict.keys():
                labelsDict[k].append([x, y])
                sizeD[k] += 1
            else:
                labelsDict[k] = [[x, y]]
                sizeD[k] = 1
    lD = {}
    sD = {}
    for k,v in labelsDict.items():
        if len(v) > 20:
            lD[k] = v
            sD[k] = sizeD[k]

    if len(lD)==0 or len(sD)==0:
        return None

    kW, kH, kS= getMeanSizeOfSinglePoints(lD, sD)  #mean

    return lD, kW, kH, kS

def getMeanSizeOfSinglePoints(labelsDict, sizeD, ratioThsld=0.5):
    sortedSizes = dict(sorted(sizeD.items(), key=lambda item: item[1]))
    sS = sortedSizes
    vS = list(sS.values())
    vSpost = vS[1:]
    vSpost.append(vS[-1]*2)
    vS = np.array(vS)
    vSpost = np.array(vSpost)

    diff = vSpost - vS
    diffRatios =  diff/np.sqrt(vSpost)
    
    candKeys = []
    ssKeys = list(sS.keys())
    isBreak = False
    for idx in range(len(diffRatios)): 
        if diffRatios[idx]<ratioThsld:
            k = ssKeys[idx]
            candKeys.append(k)
            isBreak = True
        elif isBreak:
            break
        else:
            continue
    if len(candKeys)==0:
        k = ssKeys[idx]
        candKeys.append(k)

    assert(len(candKeys)>0)
    kWHs = []
    Ss = []
    assert(len(candKeys)>0)
    for k in candKeys: 
        kW, kH = getSizeOfCompnt(labelsDict[k])
        cKey = 'n%d'%len(labelsDict[k])
        kWHs.append([kW, kH])
        Ss.append(len(labelsDict[k]))
    kWHs = np.array(kWHs)
    kW, kH = np.round(kWHs.mean(axis=0)).astype(np.uint16)
    kS = np.array(Ss).mean()

    return kW, kH, kS

def getSizeOfCompnt(v):
    v = np.array(v)
    if len(v.shape)<=1: return None
    t = min(v[:,1])
    b = max(v[:,1])
    l = min(v[:,0])
    r = max(v[:,0])
    w = r-l+1
    h = b-t+1
    return w,h

def binarize(img, bk=7, isShow=False):
    blurred = cv2.GaussianBlur(img, (bk, bk), 0)
    threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return threshold

def getRadiusCentsByCluster(pxs, N, randSeed=None):
    kmeans = KMeans(n_clusters=N, n_init=3, random_state=randSeed)
    kmeans.fit(pxs)
    lbs = kmeans.labels_
    ccs = kmeans.cluster_centers_
    rs = []

    for n in range(N):
        lb = pxs[lbs==n]
        cc = ccs[n]
        diff_ = lb-cc
        norms = LA.norm(diff_, axis=1)
        r = norms.max()
        sl,sr,st,sb = lb[:,0].min(),lb[:,0].max(),lb[:,1].min(),lb[:,1].max(),
        rs.append([r,sl,sr,st,sb])

    return rs, ccs

def getAlignParam(ccPxs, rs, ccs):
    cl = ccPxs[:,0].min()
    cr = ccPxs[:,0].max()
    ct = ccPxs[:,1].min()
    cb = ccPxs[:,1].max()

    rm = np.array(rs)[:,0].max()

    dl = ccs[:,0].min()-rm
    dr = ccs[:,0].max()+rm
    dt = ccs[:,1].min()-rm
    db = ccs[:,1].max()+rm

    l = int(min(cl,dl))-1
    r = int(max(cr,dr))+1
    t = int(min(ct,dt))-1
    b = int(max(cb,db))+1
    align = (l,r,t,b)

    return align 

def drawConnComp(pxs, align):
    l,r,t,b = align
    pxs = pxs.copy()
    pxs -= np.array([l,t])
    ovis = initialCanvas(align)
    ovis[pxs[:,1], pxs[:,0]] = True
    return ovis 

def initialCanvas(align):
    l,r,t,b = align
    w = r-l+1
    h = b-t+1
    cv = np.full((h,w), False)
    return cv

def drawCluster(rs, ccs, m, align):
    l,r,t,b = align
    ccs = np.array(ccs) - np.array([l,t])
    revis = initialCanvas(align)
    cv = revis.copy()
    assert(len(rs)==len(ccs))
    for idx in range(len(ccs)):
        r0,sl,sr,st,sb = rs[idx]
        cc = ccs[idx,:]
        if r0>1: drawMark(cv, cc, r0, m)
        else: continue #skip too small clusters
        revis = revis|cv

    return revis

def drawMark(cv, cc, r0, m):
    h,w = cv.shape
    mark,r,d = makeMark(r0,m,w,h)
    cy,cx = tuple(np.round(cc).astype(np.int16))
    cv[cx-r:cx-r+d, cy-r:cy-r+d] = mark

def makeMark(r0, m, w, h):
    #print('r0:', r0)
    assert(r0>1)
    r2 = int(np.round(2*r0))
    d = r2 if r2%2==1 else r2+1
    r = int(d/2)

    if m == 's':
        d = int(d/np.sqrt(2))+1
        d = d if d%2==1 else d+1
        r = int(d/2)

    k = '%s_%d_%d_%d_%d'%(m,d,r,w,h)
    if k in markCache.keys():
        return markCache[k], r, d

    halfd = int(d//2)
    bb = max(2,int(d//8))#border
    mark = None
    if m == 'D':
        v = np.abs(np.arange(d)-r)
        mark = np.add.outer(v,v)<=r
    if m == 'D_':
        v = np.abs(np.arange(d)-r)
        addA = np.add.outer(v,v)
        mark1 = addA<=r
        mark2 = addA<=r-bb
        mark = np.logical_xor(mark1, mark2)
    elif m == 'o':
        v = (np.arange(d)-r)**2
        mark = np.add.outer(v, v) < r**2
        mark[0, r] = True
        mark[d-1, r] = True
        mark[r, 0] = True
        mark[r, d-1] = True
    elif m == 'o_':
        v = (np.arange(d)-r)**2
        addA = np.add.outer(v, v)
        mark1 = addA < r**2
        mark2 = addA < (r-bb)**2
        mark = np.logical_xor(mark1, mark2)
    elif m == 's':
        mark = np.full((d,d), True)
    elif m == 's_':
        mark = np.full((d,d), True)
        mark[2:d-bb, 2:d-bb] = False
    elif m == '^':
        mark = np.full((d,d), False)
        for i in range(d):
            mark[i, (d-i)//2:(d+i)//2] = True
    elif m == '^_':
        mark = np.full((d,d), False)
        for i in range(d):
            mark[i, (d-i)//2:(d+i)//2] = True
        for i in range(bb,d-bb):
            mark[i, (d+bb*2-i)//2:(d-bb*2+i)//2] = False 
    elif m == 'v':
        mark = np.full((d,d), False)
        for i in range(d):
            mark[i, r-(d-i)//2:r+(d-i)//2] = True
    elif m == 'v_':
        mark = np.full((d,d), False)
        for i in range(d):
            mark[i, r-(d-i)//2:r+(d-i)//2] = True
        for i in range(bb,d-bb):
            mark[i, r-(d-bb*2-i)//2:r+(d-bb*2-i)//2] = False 
    elif m == '6':
        mark = np.full((d,d), False)
        for i in range(int(d*0.8)):
            mark[i, (d-i)//2:(d+i)//2] = True
    elif m == '7':
        mark = np.full((d,d), False)
        for i in range(int(d*0.8)):
            mark[i, r-(d-i)//2:r+(d-i)//2] = True
    elif m == '+':
        mark = np.full((d,d), False)
        mark[:, halfd-1:halfd+1] = True
        mark[halfd-1:halfd+1, :] = True
    elif m == '*':
        mark = np.full((d,d), False)
        for i in range(1, d-1):
            for j in range(i,d-1):
                if np.abs(i-j) < bb or (d-np.abs(i-j)) < bb:
                    mark[i-1:i+1,j-1:j+1] = True
                    mark[i-1:i+1,d-j-1:d-j+1] = True

    markCache[k] = mark
    return mark, r, d

def calcSymmetricDiff(s1, s2):
    symDiff = ~(s1&s2) & (s1|s2)
    return symDiff.sum() 

def getConnCompns(img):
    assert(img is not None)
    num, labels = cv2.connectedComponents(img)
    lD = {}
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            if not img[y,x]: continue
            k = labels[y][x]
            if k in lD.keys():
                lD[k].append([x, y])
            else:
                lD[k] = [[x, y]]
    return lD

def getConnCompns(img):
    assert(img is not None)
    num, labels = cv2.connectedComponents(img)
    lD = {}
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            if not img[y,x]: continue
            k = labels[y][x]
            if k in lD.keys():
                lD[k].append([x, y])
            else:
                lD[k] = [[x, y]]
    return lD

