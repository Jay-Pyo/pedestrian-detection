ICV21_FinalProject

# Environment
- python3.6.9
- GCC 8.3.0 on linux

# Pipeline
1. make directory as below
  (root)
    - *.py
    - README.txt
    - INRIAPerson
      - 70X134H96/*
      - 90x160H96/*
      - Test/*
      - test_64x128_H96/*
      - Train/*
      - train_64x128_H96/*

2. execute 'gen_data.py'

3. execute train/test (choose one of the below)
- execute 'train1.py' and then 'test1.py' for '(1)Hog'
- execute 'train2.py' and then 'test2.py' for '(2)Cascade'
- execute 'train3.py' and then 'test3.py' for '(3)LBP'

4. check results
- './test_result.txt'
  - from report_accuracy (utils.py)
  - tpr, fpr, tnr, fnr about 64x128 test images
- './ROC.png'
  - from evaluate_detections (utils.py)
  - ROC curve about full-size test images
- './precision-recall.png'
  - from evaluate_detections (utils.py)
  - precision-recall curve about full-size test images
- './result-image/fig-{input_file_name}.png'
  - from visualize_detections_by_image (utils.py)
  - on positive(= human) test images
