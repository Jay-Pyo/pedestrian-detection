import cv2
import pathlib
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import pickle
from tqdm import tqdm
from tools import *
from utils import *

def eval_64x128():
    desc_list, label_list = [], []
    for f, img, label in tqdm(test_64x128_dir):
        desc = np.reshape(model.descriptor(img), (1, -1))
        desc_list.append(desc)
        label_list.append(label)
    
    desc = np.vstack(desc_list)
    confidences = model.svm.predict_proba(desc)
    confidences = np.squeeze(confidences[:, 1])
    label_vector = np.array(label_list)
    report_accuracy(confidences, label_vector)
    

def eval_full(thr_conf, thr_iou, onlyPos=True):    
    stride = (16, 16)
    pyr_step = 0.9
    
    bbox_list, conf_list, id_list = [], [], []
    for f, img, label in tqdm(test_dir):
        if label == 0 and onlyPos: continue
        bbox, conf = model(img, stride, pyr_step)
        conf = np.squeeze(conf)
        
        bbox, conf = non_maximum(bbox, conf, thr_conf, thr_iou)
        
        bbox_list.extend(bbox)
        conf_list.extend(conf)
        id_list.extend([f] * len(bbox))
        
    bboxes = np.vstack(bbox_list)
    confidences = np.squeeze(np.vstack(conf_list))
    
        
    gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections = evaluate_detections(bboxes, confidences, id_list, str(test_dir.ann_path))
    print(gt_isclaimed)
    print(duplicate_detections)
    visualize_detections_by_image(bboxes, confidences, id_list, tp, fp, str(home_dir / 'test' / 'pos'), str(test_dir.ann_path))

if __name__ == '__main__':    
    home_dir = pathlib.Path(__file__).absolute().parent
    model_64x128_dir = home_dir / 'model_HOG.pkl'
    test_64x128_dir = DataDir(home_dir / 'test_64x128')
    test_dir = DataDir(home_dir / 'test')
    
    model = joblib.load(str(model_64x128_dir))
    
    ## 64x128 eval.
    eval_64x128()
        
    ## full size eval.
    eval_full(0.5, 0.2)