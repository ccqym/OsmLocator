import cv2
import time
import random
import numpy as np
import osmlocator.util as util

class OsmLocator:
    def __init__(self, setting):
        self.setting = setting
        allSupportableMarkers = {'o':'circle','o_':'circle_', 's':'square','s_':'square_', 'D':'diamond','D_':'diamond_', '^':'triangle_up','^_':'triangle_up_', 'v':'triangle_down','v_':'triangle_down_', '6':'careup', '7':'caredown', '+':'plus', '*':'star'}
        #self.supportableMarkers = 'o,o_,s,s_,D,D_,^,^_,v,v_,6,7,+,*' 
        markers = setting['supportable_markers'].split(',')
        self.supportableMarkers = {k:allSupportableMarkers[k] for k in markers}

        self.lcache = {}
        self.randSeed = None
        self.setSSF = setting['space_setting_factor']
        self.verbose = setting['verbose']

    def getMarkerName(self, m):
        return self.supportableMarkers[m]

    def getMarkerId(self, m):
        return list(self.supportableMarkers.keys()).index(m)

    #only accept binary image input
    def locate(self,bImg,zoom=1):
        ss = self.setting
        self.lcache = {}
        sgms = None
        if ss['is_auto_search_space']:
            sgms = util.getConnCompAndSingleMarkSize(bImg, zoom)
        if sgms is not None:
            connCompns, _, _, kS = sgms
            ssf = kS*ss['impact_of_recognized_marks']
            if ssf < 15:
                ssf = ss['space_setting_factor']
                print('ignoring too small single marks', kS, ssf)
        else:
            connCompns = util.getConnCompns(bImg)
            ssf = ss['space_setting_factor']
        mlocsAll = [] 
        h,w = bImg.shape
        for ccId, ccPxs in connCompns.items(): 
            if len(ccPxs) < 16: 
                print('ignore too small regions.', ccId)
                continue

            mlocs = self.optimizeBySA(ccId, ccPxs, ssf)

            if mlocs is not None and len(mlocs)>0:
                mlocsAll.extend(mlocs)

        mlocsAllD = [{'x':x/zoom, 'y':y/zoom, 'm':m} for x,y,m in mlocsAll]
        return mlocsAllD

    def newMN(self, m, n, N0):
        M0 = len(self.supportableMarkers)
        m1 = int(random.normalvariate(m, M0/6)%M0)
        n1 = max(1, int(random.normalvariate(n, N0/6)%N0))
        return m1,n1

    def optimizeBySA(self, ccId, ccPxs, ssf):
        ss = self.setting
        sms = list(self.supportableMarkers.keys())
        M0 = len(sms)
        ccpxN = len(ccPxs)
        N0 = int(max(1, ccpxN/ssf))
        m0 = None
        mkvSetpN = int(ss['gamma_markov']*np.log(N0*M0))
        ccPxs = np.array(ccPxs)

        l,m,n,d= self.objInitialize(ccId, ccPxs, N0, m0, ssf)
        best, bestPre = (l,l)
        bestSol = (m,n,d)
        if self.verbose: print('initial solution:', sms[m],n, N0)
        M0 = len(sms)
        ccpxN = len(ccPxs)

        T = T0 = N0
        Tcnt = 1
        bestStbCnt = 0
        stopCrit = ss['gamma_stop'] * np.log(N0*M0)
        while T>0.1 and bestStbCnt<stopCrit:
            for j in range(mkvSetpN):
                m1, n1 = self.newMN(m,n, N0)
                if self.verbose: print('new solution:', sms[m1],n1)
                if m1==m and n1==n: 
                    continue
                l1, d1, _ = self.objective(ccId, ccPxs, m1, n1, N0, ssf)
                if l1<l:
                    if self.verbose: print('++ accept new solution:', sms[m1],n1,sms[m],n)
                    n,m,l,d = (n1,m1,l1,d1)
                    if l1 < best: 
                        best = l
                        bestSol = (m,n,d)
                else:
                    p = min(1, np.exp((l-l1)/T))
                    r = np.random.uniform(0,1)
                    if p>r:
                        if self.verbose: print('++ accept new solution wth p:', sms[m1],n1,sms[m],n,l-l1,T,p)
                        n,m,l,d = (n1,m1,l1,d1)
                    else:
                        if self.verbose: print('-- decline new solution:', sms[m1],n1, sms[m],n,l-l1,T,p)
            T = T0 / (1.0+np.log(1+Tcnt))
            Tcnt += 1
            if best < bestPre:
                bestStbCnt = 0
            else: 
                bestStbCnt += 1
            bestPre = best

        m,n,d = bestSol
        mlocs = [[x,y,m] for x,y in d]
        return mlocs 

    def objInitialize(self, ccId, ccPxs, N0, m0, ssf):
        n = 1
        if m0 is not None:
            m = m0
            l, d, _ = self.objective(ccId, ccPxs, m, n, N0, ssf)
            return l,m,n,d

        sms = self.supportableMarkers.keys()
        l_ = 1e+10
        for m in range(len(sms)):
            l, d, _ = self.objective(ccId, ccPxs, m, n, N0, ssf)
            if l < l_: 
                l_,m_,n_,d_ = l,m,n,d

        return l_,m_,n_,d_

    def objective(self, ccId, ccPxs, m, n, N0, ssf):
        assert(n < len(ccPxs))
        ss = self.setting
        sm =  list(self.supportableMarkers.keys())[m]
        k = '%d_%s%d'%(ccId, sm, n)
        if self.verbose: print('handling:', k)
        if k in self.lcache:
            if self.verbose: print('cache restore:', k)
            return self.lcache[k]
        rs, ccs = util.getRadiusCentsByCluster(ccPxs, n, self.randSeed)
        align = util.getAlignParam(ccPxs, rs, ccs)
        ovis = util.drawConnComp(ccPxs, align)
        revis = util.drawCluster(rs, ccs, sm, align)
        diff = util.calcSymmetricDiff(ovis, revis)
        sigma = np.array(rs)[:,0].std()

        second = ss['alpha']*((n/N0)*diff + n*np.sqrt(ssf))
        third =  ss['beta']*sigma
        obj = diff + second + third
        self.lcache[k] = (obj, ccs, (diff, second, third))
        if self.verbose: print(k, N0, obj, diff, second, third)

        return obj, ccs, (diff, second, third)
