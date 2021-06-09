import cv2
import pathlib
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import pickle
from tools import *

if __name__ == '__main__':
    home_dir = pathlib.Path(__file__).absolute().parent
    model_64x128_dir = home_dir / 'model_cascade.pkl'
    
    args = get_args()
    if args.aug:
        pos_64x128_dir = ImageDir(\
        home_dir / 'train_64x128' / 'pos',\
        home_dir / 'train_64x128' / 'pos-aug-from-rotating'\
        )
        neg_64x128_dir = ImageDir(\
        home_dir / 'train_64x128' / 'neg',\
        home_dir / 'train_64x128' / 'neg-aug-from-pos'\
        )
    else:
        pos_64x128_dir = ImageDir(home_dir / 'train_64x128' / 'pos')
        neg_64x128_dir = ImageDir(home_dir / 'train_64x128' / 'neg')
    
    pos_stack = []
    neg_stack = []
    
    for _, img in pos_64x128_dir:
        pos_stack.append([img])
    
    for _, img in neg_64x128_dir:
        neg_stack.append([img])
        
    pos_stack = np.vstack(pos_stack)
    neg_stack = np.vstack(neg_stack)
    print(pos_stack.shape, neg_stack.shape)
        
    model = CascadeDetector(F_target=0.005, f_max=0.7, d_min=0.9975, svm_sample=60)
    model.train(pos_stack, neg_stack)
    
    joblib.dump(model, str(model_64x128_dir))
    print("complete.")