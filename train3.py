import cv2
import pathlib
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import pickle
from tools import *
from tqdm import tqdm

if __name__ == '__main__':
    home_dir = pathlib.Path(__file__).absolute().parent
    model_64x128_dir = home_dir / 'model_LBP.pkl'
    
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
    
    lbp = LBP(32, 3)
    stride = lbp.block_size
    
    X = []
    Y = []
    sample_weight = []
    
    for f, img in tqdm(pos_64x128_dir):
        desc = lbp(img, stride)
        desc = np.squeeze(desc)
        X.append(desc)
        Y.append(1)
        if f[:3] == 'aug':
            sample_weight.append(0.5)
        else:
            sample_weight.append(1)
    
    for f, img in tqdm(neg_64x128_dir):
        desc = lbp(img, stride)
        desc = np.squeeze(desc)
        X.append(desc)
        Y.append(0)
        if f[:4] == 'from':
            sample_weight.append(0.5)
        else:
            sample_weight.append(1)
        
    svm = SVC(kernel='linear', probability=True)
    X = np.vstack(X)
    Y = np.squeeze(np.vstack(Y))
    
    print(X.shape)
    print(Y.shape)
    
    svm.fit(X, Y, sample_weight=sample_weight)
    
    model = SimpleDetector(lbp, svm)
    
    joblib.dump(model, str(model_64x128_dir))
    print("complete.")