import cv2
import pathlib
import numpy as np
from tools import *
import random
import os, shutil
import imutils

global home_dir
global ext

def generate_train_64x128(duplicate_num = 10, sample_rate = 1):
    get_pos_dir = ImageDir(home_dir / 'INRIAPerson' / '96X160H96' / 'Train' / 'pos')
    get_neg_dir = ImageDir(home_dir / 'INRIAPerson' / 'Train' / 'neg')
    save_pos_dir = home_dir / 'train_64x128' / 'pos'
    save_neg_dir = home_dir / 'train_64x128' / 'neg'
    if os.path.exists(save_pos_dir):
        shutil.rmtree(save_pos_dir)
    if os.path.exists(save_neg_dir):
        shutil.rmtree(save_neg_dir)
    os.makedirs(save_pos_dir)
    os.makedirs(save_neg_dir)
    
    for f, img in get_pos_dir:
        if not random.uniform(0, 1) <= sample_rate: continue
        path = save_pos_dir / '{0}.{1}'.format(f[:-4], ext)
        crop = img[16 : -16, 16 : - 16]
        cv2.imwrite(str(path), crop)
        
        assert crop.shape[:2] == (128, 64)
    
    for f, img in get_neg_dir:
        if not random.uniform(0, 1) <= sample_rate: continue
        for i in range(duplicate_num):
            x_max = img.shape[1] - 64
            y_max = img.shape[0] - 128
            x = np.random.randint(0, x_max)
            y = np.random.randint(0, y_max)
            
            crop = img[y : y+128, x : x + 64]
            path = save_neg_dir / '{0}{1}.{2}'.format(f[:-4], i, ext)
            cv2.imwrite(str(path), crop)
            
            assert crop.shape[:2] == (128, 64)

def generate_test_64x128(sample_rate = 1):
    get_pos_dir = ImageDir(home_dir / 'INRIAPerson' / '70X134H96' / 'Test' / 'pos')
    get_neg_dir = ImageDir(home_dir / 'INRIAPerson' / 'Test' / 'neg')
    save_pos_dir = home_dir / 'test_64x128' / 'pos'
    save_neg_dir = home_dir / 'test_64x128' / 'neg'
    if os.path.exists(save_pos_dir):
        shutil.rmtree(save_pos_dir)
    if os.path.exists(save_neg_dir):
        shutil.rmtree(save_neg_dir)
    os.makedirs(save_pos_dir)
    os.makedirs(save_neg_dir)
    
    for f, img in get_pos_dir:
        if not random.uniform(0, 1) <= sample_rate: continue
        path = save_pos_dir / '{0}.{1}'.format(f[:-4], ext)
        crop = img[3 : -3, 3 : - 3]
        cv2.imwrite(str(path), crop)
        
        assert crop.shape[:2] == (128, 64)
    
    for f, img in get_neg_dir:
        if not random.uniform(0, 1) <= sample_rate: continue
        for i in range(10):
            x_max = img.shape[1] - 64
            y_max = img.shape[0] - 128
            x = np.random.randint(0, x_max)
            y = np.random.randint(0, y_max)
            
            crop = img[y : y+128, x : x + 64]
            path = save_neg_dir / '{0}{1}.{2}'.format(f[:-4], i, ext)
            cv2.imwrite(str(path), crop)
            
            assert crop.shape[:2] == (128, 64)
            
def generate_test(sample_rate = 1):
    get_pos_dir = ImageDir(home_dir / 'INRIAPerson' / 'Test' / 'pos')
    get_ann_dir = AnnotationDir(home_dir / 'INRIAPerson' / 'Test' / 'annotations') 
    get_neg_dir = ImageDir(home_dir / 'INRIAPerson' / 'Test' / 'neg')
    save_pos_dir = home_dir / 'test' / 'pos'
    save_ann_dir = home_dir / 'test' / 'pos_label.txt'
    save_neg_dir = home_dir / 'test' / 'neg'
    if os.path.exists(save_pos_dir):
        shutil.rmtree(save_pos_dir)
    if os.path.exists(save_neg_dir):
        shutil.rmtree(save_neg_dir)
    os.makedirs(save_pos_dir)
    os.makedirs(save_neg_dir)
    
    sample_f = []
    
    for f, img in get_pos_dir:
        if random.uniform(0, 1) <= sample_rate:
            sample_f.append(f[:-4])
        else: continue
        path = save_pos_dir / '{0}.{1}'.format(f[:-4], ext)
        cv2.imwrite(str(path), img)
    
    for f, img in get_neg_dir:
        if random.uniform(0, 1) <= sample_rate:
            sample_f.append(f[:-4])
        else: continue
        path = save_neg_dir / '{0}.{1}'.format(f[:-4], ext)
        cv2.imwrite(str(path), img)    
    
    save_ann_file = open(str(save_ann_dir), 'w')
    for f, gt_num, bbox_pos_list in get_ann_dir:
        if not f[:-4] in sample_f: continue
        for bbox_pos in bbox_pos_list:
            save_ann_line = '{0}.{1} {2} {3} {4} {5}\n'.format(f[:-4], ext, bbox_pos[0], bbox_pos[1], bbox_pos[2], bbox_pos[3])
            save_ann_file.write(save_ann_line)
        assert gt_num == len(bbox_pos_list)
    save_ann_file.close()
    
def neg_augmentation_from_pos(extract_num = 10):
    MAXIMUM_TRIAL = extract_num * 10
    get_pos_dir = ImageDir(home_dir / 'INRIAPerson' / 'Train' / 'pos')
    get_ann_dir = AnnotationDir(home_dir / 'INRIAPerson' / 'Train' / 'annotations')
    save_neg_dir = home_dir / 'train_64x128' / 'neg-aug-from-pos'
    if os.path.exists(save_neg_dir):
        shutil.rmtree(save_neg_dir)
    os.makedirs(save_neg_dir)
    
    for f, gt_num, bbox_pos_list in get_ann_dir:
        img = cv2.imread(str(get_pos_dir.root_list[0] / f))
        crop_box_list = []
        for i, (xmin, ymin, xmax, ymax) in enumerate(bbox_pos_list):
            if xmax - xmin > 64 * 3 and ymax - ymin > 128 * 3:
                crop_lt = img[ymin:ymin+128, xmin:xmin+64]
                crop_rt = img[ymax-128:ymax, xmax-64:xmax]
                crop_lb = img[ymin:ymin+128, xmin:xmin+64]
                crop_rb = img[ymax-128:ymax, xmax-64:xmax]
                path_lt = save_neg_dir / 'aug_from_pos_{0}{1}lt.{2}'.format(f[:-4], i, ext)
                path_rt = save_neg_dir / 'aug_from_pos_{0}{1}rt.{2}'.format(f[:-4], i, ext)
                path_lb = save_neg_dir / 'aug_from_pos_{0}{1}lb.{2}'.format(f[:-4], i, ext)
                path_rb = save_neg_dir / 'aug_from_pos_{0}{1}rb.{2}'.format(f[:-4], i, ext)
                cv2.imwrite(str(path_lt), crop_lt)
                cv2.imwrite(str(path_rt), crop_rt)
                cv2.imwrite(str(path_lb), crop_lb)
                cv2.imwrite(str(path_rb), crop_rb)
        
        for _ in range(MAXIMUM_TRIAL):
            w = random.randrange(64, 128)
            h = w * 2
            if img.shape[1] <= w or img.shape[0] <= h: continue
            x0 = random.randrange(0, img.shape[1]-w)
            y0 = random.randrange(0, img.shape[0]-h)
            crop_box = (w, h, x0, y0)
            flag = False
            
            for xmin, ymin, xmax, ymax in bbox_pos_list:
                if (xmin < x0 < xmax and ymin < y0 < ymax) or\
                (xmin < x0+w < xmax and ymin < y0 < ymax) or\
                (xmin < x0 < xmax and ymin < y0+h < ymax) or\
                (xmin < x0+w < xmax and ymin < y0+h < ymax):
                    flag = True
                    break
            
            for crop_box_ in crop_box_list:
                if crop_box == crop_box_:
                    flag = True
                    break
                
            
            if not flag:
                crop_box_list.append(crop_box)
                if len(crop_box_list) >= extract_num: break
                
        for i, (w, h, x0, y0) in enumerate(crop_box_list):
            crop = img[y0 : y0+h, x0 : x0+w]
            crop = cv2.resize(crop, dsize=(64, 128), interpolation=cv2.INTER_AREA)
            path = save_neg_dir / 'aug_from_pos_{0}{1}.{2}'.format(f[:-4], i, ext)
            cv2.imwrite(str(path), crop)
    
def pos_augmentation_from_rotating(duplicate_num = 4):
    get_dir = ImageDir(home_dir / 'INRIAPerson' / '96X160H96' / 'Train' / 'pos')
    save_dir = home_dir / 'train_64x128' / 'pos-aug-from-rotating'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    
    for f, img in get_dir:
        for _ in range(duplicate_num):
            path = save_dir / '{0}.{1}'.format(f[:-4], ext)
            rotate = imutils.rotate(img, random.uniform(-5, 5))
            crop = rotate[16 : -16, 16 : - 16]
            cv2.imwrite(str(path), crop)
            
            assert crop.shape[:2] == (128, 64)
    
if __name__ == '__main__':
    global home_dir
    global ext
    home_dir = pathlib.Path(__file__).absolute().parent
    ext = 'jpg'
    
    generate_train_64x128(duplicate_num = 10, sample_rate = 1)
    print("complete.")
    generate_test_64x128(sample_rate = 1)
    print("complete.")
    generate_test(sample_rate = 0.3)
    print("complete.")
    neg_augmentation_from_pos(extract_num = 20)
    print("complete.")
    pos_augmentation_from_rotating(duplicate_num = 1)
    print("complete.")