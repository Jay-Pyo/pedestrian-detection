import cv2
import pathlib
import numpy as np
from sklearn.svm import SVC
import os
import re
import math, random
import argparse
from skimage.feature import local_binary_pattern as LBPDescriptor

class ImageDir:
    def __init__(self, *root):
        self.root_list = root
        
    def __iter__(self):
        for root in self.root_list:
            for f in os.listdir(root):
                path = root / f
                img = cv2.imread(str(path))
                yield f, img
            
class AnnotationDir:
    def __init__(self, root):
        self.root = root
        # Image filename : "Train/pos/crop001001.png"
        self.file_name_pattern = 'Image filename : ".*\/(.*)"'
        # Objects with ground truth : 3 { "PASperson" "PASperson" "PASperson" }
        self.gt_num_pattern = 'Objects with ground truth : (\d+) { .* }'
        # Bounding box for object 3 "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (148, 179) - (290, 641)
        self.bbox_pos_pattern = 'Bounding box for object \d+ ".*" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)'
        
    def __iter__(self):
        for f in os.listdir(self.root):
            fn, gt_num, bbox_pos_list = '', 0, []
            
            path = self.root / f
            with open(str(path), 'r', encoding='utf-8', errors='replace') as ann_file:
                for line in ann_file:
                    match0 = re.findall(self.file_name_pattern, line)
                    match1 = re.findall(self.gt_num_pattern, line)
                    match2 = re.findall(self.bbox_pos_pattern, line)
                    for m in match0:
                        fn = m
                    for m in match1:
                        gt_num = int(m)    
                    for m in match2:
                        bbox_pos_list.append(( int(m[0]),int(m[1]),int(m[2]),int(m[3]), ))
            yield fn, gt_num, bbox_pos_list   
            

class DataDir:
    def __init__(self, root):
        self.root = root
        self.pos_dir = ImageDir(root / 'pos')
        self.neg_dir = ImageDir(root / 'neg')
        self.ann_path = root / 'pos_label.txt'
        
    def __iter__(self):
        for f, img in self.pos_dir:
            yield f, img, 1
        
        for f, img in self.neg_dir:
            yield f, img, 0

class HOG:
    def __init__(self, win_size=(64,128), block_size=(16,16), block_stride=(8,8), cell_size=(8,8), n_bins=9):
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.n_bins = n_bins
        
    def __call__(self, img, stride=None, locations=None):
        hog = cv2.HOGDescriptor(self.win_size, self.block_size, self.block_stride, self.cell_size, self.n_bins)
        if locations:
            return hog.compute(img, locations=locations)
        elif stride:
            return hog.compute(img, winStride=stride)
        else:
            return hog.compute(img, winStride=self.block_stride)
            
class LBP:
    def __init__(self, np, radius, block_size=(16,16), block_stride=(16,16)):
        self.np = np
        self.radius = radius
        self.block_size = block_size
        self.block_stride = block_stride
    
    def __call__(self, img, stride=None):
        if not stride: stride = self.block_stride
        img = np.mean(img, -1).astype('uint8')
        lbp = LBPDescriptor(img, self.np, self.radius, method='uniform')
        r = (img.shape[0] - 128) // stride[1] + 1
        c = (img.shape[1] - 64) // stride[0] + 1
        r_blocks = (64 - self.block_size[1]) // self.block_stride[1] + 1
        c_blocks = (128 - self.block_size[0]) // self.block_stride[0] + 1                        
        feature = np.zeros((r, c, r_blocks*c_blocks*(self.np+2)))
        for i in range(r):
            for j in range(c):
                x0 = j * stride[0]
                y0 = i * stride[1]
                lbp_window = lbp[y0 : y0+128, x0 : x0+64]
                feature[i][j] = self.compute(lbp_window)
        return feature.reshape((-1, 1))
        
    def compute(self, lbp_window):
        feature_list = []    
        for i in range(0, 128-self.block_size[1]+self.block_stride[1], self.block_stride[1]):
            for j in range(0, 64-self.block_size[0]+self.block_stride[0], self.block_stride[0]):                        
                lbp_block = lbp_window[i : i+self.block_size[1], j : j+self.block_size[0]]
                hist, _ = np.histogram(lbp_block.ravel(), bins=range(0, self.np+3))
                feature_list.extend(hist / np.sum(hist))
        return np.array(feature_list)                                                                                
                
class SimpleDetector:
    def __init__(self, descriptor, svm):
        self.descriptor = descriptor
        self.svm = svm
        
    def run_on_pyr_level(self, pyr_img, stride, pyr_step, step):
        desc = self.descriptor(pyr_img, stride)
        r = (pyr_img.shape[0] - 128) // stride[1] + 1
        c = (pyr_img.shape[1] - 64) // stride[0] + 1
        desc = np.reshape(desc, (r * c, -1))
        
        bbox = np.zeros((r * c, 4))
        for i in range(r):
            for j in range(c):
                xmin = j * stride[0]
                ymin = i * stride[1]
                xmax = xmin + 64
                ymax = ymin + 128
                
                xmin = int(xmin / pyr_step**step)
                ymin = int(ymin / pyr_step**step)
                xmax = int(xmax / pyr_step**step)
                ymax = int(ymax / pyr_step**step)
                
                bbox[i * c + j] = [xmin, ymin, xmax, ymax]
                
        return bbox, desc
        
    def __call__(self, img, stride, pyr_step):
        step, image = 0, img.copy()
        bbox, desc = [], []
        
        while image.shape[1] >= 64 and image.shape[0] >= 128:
            bbox_, desc_ = self.run_on_pyr_level(image, stride, pyr_step, step)
            bbox.append(bbox_)
            desc.append(desc_)
            step += 1
            image = cv2.resize(img, None, fx=pyr_step**step, fy=pyr_step**step, interpolation=cv2.INTER_AREA)
        
        bbox = np.concatenate(bbox)
        desc = np.concatenate(desc)
        conf = self.svm.predict_proba(desc)[:, 1]
        
        return bbox, conf
        
class CascadeWeak:
    def __init__(self, x0, y0, w, h):
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        self.hog = HOG(win_size=(w,h), block_size=(w,h), block_stride=(w,h), cell_size=(w//2,h//2), n_bins=9)
        self.svm = SVC(kernel='linear')
        
    def __call__(self, window):
        return self.predict(window)
        
    @classmethod
    def new_random(cls):
        # 0> 1:1
        # 1> 1:2
        # 2> 2:1
        mode = random.randint(0, 2)
        if mode == 0:
            w = random.randrange(12, 64, 2)
            h = w
        elif mode == 1:
            w = random.randrange(12, 64, 2)
            h = w * 2
        elif mode == 2:
            h = random.randrange(12, 32, 2)
            w = h * 2
        if w * h < 1000:
            x0 = random.randrange(0, 64 - w, 4)
            y0 = random.randrange(0, 128 - h, 4)
        else:
            x0 = random.randrange(0, 64 - w, 8)
            y0 = random.randrange(0, 128 - h, 8)    
        return cls(x0, y0, w, h)
        
    def predict(self, window):
        desc = self.hog(window, locations=[(self.x0,self.y0)])
        desc = np.reshape(desc, (1, -1))
        h = self.svm.predict(desc)
        return np.squeeze(h)
    
    def train(self, img_stack, Y, sample_weight):
        desc_list = []
        for img in img_stack:
            desc = self.hog(img, locations=[(self.x0,self.y0)])
            desc = np.squeeze(desc)
            desc_list.append([desc])
        X = np.vstack(desc_list)
        
        self.svm.fit(X, Y, sample_weight=sample_weight)
        
        pred = self.svm.predict(X)
        e = (pred != Y).astype(int)
        err = np.sum(e * sample_weight) / np.sum(sample_weight)
        return pred, e, err
        
class CascadeStrong:
    def __init__(self):
        self.h_list = []
        self.alpha_list = []
        self.thr = 0
        
    def __call__(self, window):
        sum = 0
        for h, alpha in zip(self.h_list, self.alpha_list):
            sum += alpha * h(window)
        return sum
    
    def predict(self, window):
        return int(self(window) >= self.thr)
        
    def confidence(self, window):
        gamma = -math.log(2)/math.log(self.thr / np.sum(self.alpha_list))
        assert gamma > 0
        return (self(window) / np.sum(self.alpha_list)) ** gamma
        
    def addWeak(self, weak, e, err, sample_weight):
        eps = 1e-6
        beta = err / (1 - err + eps)
        alpha = -math.log(beta)
        
        self.h_list.append(weak)
        self.alpha_list.append(alpha)
        return sample_weight * np.power(beta, 1 - e)
        
    def train(self, pos_stack, neg_stack, f_max, d_min, svm_sample):
        fpr = 1
        Y = np.concatenate((np.ones((pos_stack.shape[0])), np.zeros((neg_stack.shape[0]))))
        sample_weight = np.zeros(Y.shape)
        sample_weight[Y == 0] = 0.5 / np.sum(Y == 0)
        sample_weight[Y == 1] = 0.5 / np.sum(Y == 1)
        h = np.zeros(Y.shape)
        
        while fpr > f_max:
            weak_best = None
            pred_best = None
            e_best = None
            err_best = float('inf')
            
            img_stack = np.concatenate((pos_stack, neg_stack))
            sample_weight = sample_weight * np.mean(1/sample_weight)
            #print('[sample_weight]', sample_weight)
            
            for _ in range(svm_sample):
                weak = CascadeWeak.new_random()
                pred, e, err = weak.train(img_stack, Y, sample_weight)
                if err_best > err:
                    weak_best = weak
                    pred_best = pred
                    e_best = e
                    err_best = err
                print('[weak-classifier] pos=({0},{1})-({2},{3}), err={4}'.format(weak.x0, weak.y0, weak.x0+weak.w, weak.y0+weak.h, err))
            
            sample_weight = self.addWeak(weak_best, e_best, err_best, sample_weight)
            h += self.alpha_list[-1] * pred_best
            #print('h', h[:296])
            #print('Y', Y[:296])
            #print('eBest', e_best[:296])
            
            order = np.argsort(-h)
            conf = h[order]
            label = Y[order]
            cum_label = np.cumsum(label)
            idx = np.where(cum_label == int(cum_label[-1] * d_min))[0][0]
            
            self.thr = conf[idx]
            #print("[thr]", self.thr)
            fpr = np.sum(np.logical_and(label == 0, conf >= self.thr)) / np.sum(label == 0)
            print("[fpr]", fpr)
        
        return img_stack[np.logical_and(h > self.thr, Y == 0)], fpr
        
class CascadeDetector:
    def __init__(self, F_target, f_max, d_min, svm_sample):
        self.F_target = F_target
        self.f_max = f_max
        self.d_min = d_min
        self.svm_sample = svm_sample
        self.cascade_list = []
        
    def run_on_window_level(self, window):
        for cascade in self.cascade_list[:-1]:
            if cascade.predict(window) == 0: return 0
        return self.cascade_list[-1].confidence(window)
        
    def run_on_pyr_level(self, pyr_img, stride, pyr_step, step):
        r = (pyr_img.shape[0] - 128) // stride[1] + 1
        c = (pyr_img.shape[1] - 64) // stride[0] + 1
        
        bbox = np.zeros((r * c, 4))
        conf = np.zeros((r * c, 1))
        for i in range(r):
            for j in range(c):
                xmin = j * stride[0]
                ymin = i * stride[1]
                xmax = xmin + 64
                ymax = ymin + 128
                
                window = pyr_img[ymin:ymax, xmin:xmax]
                conf[i * c + j] = self.run_on_window_level(window)
                
                xmin = int(xmin / pyr_step**step)
                ymin = int(ymin / pyr_step**step)
                xmax = int(xmax / pyr_step**step)
                ymax = int(ymax / pyr_step**step)
                
                bbox[i * c + j] = [xmin, ymin, xmax, ymax]
                
        return bbox, conf
        
    def __call__(self, img, stride, pyr_step):
        step, image = 0, img.copy()
        bbox, conf = [], []
        
        while image.shape[1] >= 64 and image.shape[0] >= 128:
            bbox_, conf_ = self.run_on_pyr_level(image, stride, pyr_step, step)
            bbox.append(bbox_)
            conf.append(conf_)
            step += 1
            image = cv2.resize(img, None, fx=pyr_step**step, fy=pyr_step**step, interpolation=cv2.INTER_AREA)
        
        bbox = np.concatenate(bbox)
        conf = np.concatenate(conf)
        
        return bbox, np.squeeze(conf)
    
    def train(self, pos_stack, neg_stack):
        i, F = 0, 1
        while F > self.F_target:
            print('cascade #{0} building start.'.format(i))
            i += 1
            cascade_i = CascadeStrong()
            neg_stack, fpr = cascade_i.train(pos_stack, neg_stack, self.f_max, self.d_min, self.svm_sample)
            self.cascade_list.append(cascade_i)
            print('F = {0} * {1} -> {2}'.format(F, fpr, F*fpr))
            F *= fpr
        
    def print_summary(self):
        for i, cascade in enumerate(self.cascade_list):
            print("cascade #{0} with {1} weak classifiers".format(i, len(cascade.h_list)))
        
                
def calc_iou(bb, bbgt):
    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]),
          min(bb[3], bbgt[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1

    if (iw > 0) and (ih > 0):
        ua = (bb[2]-bb[0]+1) * (bb[3]-bb[1]+1) +\
             (bbgt[2]-bbgt[0]+1) * (bbgt[3]-bbgt[1]+1) -\
             iw*ih
        ov = iw*ih / ua
        return ov
    else: return 0
    
def calc_ioa_iob(bba, bbb):
    bi = [max(bba[0], bbb[0]), max(bba[1], bbb[1]), min(bba[2], bbb[2]),
          min(bba[3], bbb[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1

    if (iw > 0) and (ih > 0):
        aa = (bba[2]-bba[0]+1) * (bba[3]-bba[1]+1)
        ab = (bbb[2]-bbb[0]+1) * (bbb[3]-bbb[1]+1)
        oa = iw*ih / aa
        ob = iw*ih / ab
        return oa, ob
    else: return 0, 0
        
def non_maximum(bbox, conf, thr_conf, thr_iou, thr_contain=0.9):
    filter_conf = conf > thr_conf 
    bbox = bbox[filter_conf]
    conf = conf[filter_conf]
    
    sort_order = np.argsort(-conf)
    bbox = bbox[sort_order]
    conf = conf[sort_order]
    
    filter_iou = np.zeros(conf.shape, dtype='bool')
    
    for i in range(len(bbox)):
        bbi = bbox[i]
        flag = False
        for bbj in bbox[filter_iou]:
            iou = calc_iou(bbi, bbj)
            if calc_iou(bbi, bbj) > thr_iou:
                flag = True
                break
        if not flag:
            filter_iou[i] = True
    
    bbox = bbox[filter_iou]
    conf = conf[filter_iou]
    
    filter_contain = np.ones(conf.shape, dtype='bool')
    flag_contain = True
    
    while flag_contain:
        flag_contain = False
        for i in range(len(bbox)):
            if not filter_contain[i]: continue
            bbi = bbox[i]
            for j in range(i):
                if not filter_contain[j]: continue
                bbj = bbox[j]
                
                ioi, ioj = calc_ioa_iob(bbi, bbj)
                if ioi > thr_contain and ioj > thr_contain:
                    bbox[i] = [min(bbi[0],bbj[0]), min(bbi[1],bbj[1]), max(bbi[2],bbj[2]), max(bbi[3],bbj[3])]
                    conf[i] = max(conf[i], conf[j])
                    filter_contain[j] = False
                    flag_contain = True
                elif ioj > thr_contain:
                    conf[i] = max(conf[i], conf[j])
                    filter_contain[j] = False
                    flag_contain = True
                elif ioi > thr_contain:
                    conf[j] = max(conf[i], conf[j])
                    filter_contain[i] = False
                    flag_contain = True
    
    return bbox[filter_contain], conf[filter_contain]
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', type=bool, nargs='?', const=True, default=False, help='using augmentation?')
    args = parser.parse_args()
    
    return args