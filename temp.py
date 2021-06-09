import cv2
import pathlib
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import pickle
from tools import *
import random

if False:  
    hog = cv2.HOGDescriptor((24, 48), (24, 48), (64, 128), (12, 24), 9)
    img = cv2.imread('./INRIAPerson/Test/pos/crop001512.png')
    
    img1 = img[:128, :64]
    desc1 = hog.compute(img1, winStride=(64,64), padding=(0,1))
    print(desc1.shape)
    desc1 = np.squeeze(desc1)
    desc1 = np.reshape(desc1, (-1, 36))
    print(desc1.shape)
    print(desc1)
    
    img2 = img[:48, :24]
    desc2 = hog.compute(img2)
    desc2 = np.squeeze(desc2)
    print(desc2.shape)
    print(desc2)
    
if False:
    model = SVC(kernel='linear', probability=True)
    X = np.array([[50,200,250],[243,250,240],[100,150,200],[120,42,0]])
    Y = np.array([[1], [0], [1], [1]])
    
    sw=None
    #sw = [1,1,1,1]
    
    model.fit(X, Y, sample_weight=sw)
    
    pred = model.predict_proba(X)
    print(sw)
    print(pred)
    
if False:
    home_dir = pathlib.Path(__file__).absolute().parent
    img_dir = ImageDir(home_dir / 'train_64x128' / 'pos')

if True:
    home_dir = pathlib.Path(__file__).absolute().parent
    test_dir = DataDir(home_dir / 'test')
    
    bboxes = np.load('bboxes.npy')
    confidences = np.load('confidences.npy')
    with open('id_list.txt', 'rb') as f:
        id_list = pickle.load(f)
        
    gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections = evaluate_detections(bboxes, confidences, id_list, str(test_dir.ann_path))