## Assignment 3: Small object detection with YOLO

 
# Prerequisite
 pip install ultralytics opencv-python numpy torch torchvisior torchaudio

 ![Mapillary Traffic Sign Database](./pic/MTSD.png) from https://www.mapillary.com/dataset/trafficsign

 # TO DO
 First, download all the zip files in the above picture (except training.zip) and unzip them
 
 Then name the unzip files accordingly: train0 -> images1, train1 -> images2, train2 -> images3, val.zip -> val
 
 create a folder named tests and throw them all in there

 then run the following programs in order

 python [filename]

 ![dataconversion](./dataconversion.py)

 ![SplitTest](./SplitTest.py)

 ![EvalPretrainModel](./Evaluation.py) (optional)

 ![Train](./Train.py)

 ![Eval](./NewModelEval.py)