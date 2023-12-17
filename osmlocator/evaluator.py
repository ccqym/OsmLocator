import numpy as np
import scipy.spatial.distance as dist
import scipy.optimize as opt

def evaluateXY(gts, pts, shape):
    gts = [[s['x'],s['y']] for s in gts]
    pts = [[s['x'],s['y']] for s in pts]
    return evaluate(gts, pts, shape)

def evaluate(gts, pts, shape):
    assScores = {}
    for lmd in [1,5,10]:
        assScore = calcAssignBasedScore(gts, pts, lmd, shape)
        assScores[str(lmd)] = assScore 
    scores = {'ass_score':assScores}
    return scores

def getCostMat(gts, pts, lamda):
    maxN = max(len(pts), len(gts))
    gts = np.array(gts)
    pts = np.array(pts)
    icov = np.linalg.inv(np.cov(gts.T)).T
    ppdist = dist.cdist(pts, gts, metric='mahalanobis', VI=icov)/lamda
    distsM = np.minimum(1, ppdist)
    h,w = distsM.shape
    costM = np.full((maxN, maxN), 1.0)
    costM[0:h, 0:w] = distsM
    return costM

def calcAssignBasedScore(gts, pts, lamda=1.0, shape=None):
    if len(gts)==0:
        print('[warn] empty gts:')
        return 1.0 if len(pts)==0 else 0.0

    cov = np.cov(np.array(gts).T)
    dovdet = np.linalg.det(cov)
    if len(gts) == 1 or dovdet<1E-3:
        print('[warn] determinant is zero or len of gts is 1')
        if len(pts) < 1:
            return 0.0
        scores = []
        lgt = np.linalg.norm(shape[0:2])
        gts = np.array(gts)
        for pt in pts:
            dis = np.linalg.norm(gts-np.array([pt]), axis=1).min()
            score = 1-dis/lgt
            scores.append(score)
        fs = np.array(scores).mean()
        return fs

    if len(pts)==0:
        print('[warn] empty pts:')
        return 0.0

    costM = getCostMat(gts, pts, lamda)
    rowIdx, colIdx = opt.linear_sum_assignment(costM)
    costLSA = costM[rowIdx, colIdx]
    cost = costLSA.sum() 
    score = 1.0 - (cost/max(len(pts), len(gts)))
    return score
