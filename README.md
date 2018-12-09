# CAM2_dark_image_classification

This repository is made by Yuefan Fu, for classify dark images.
If you have any question please contact fu194@purdue.edu 

Program:
  cnnmodel.py:
    the convolutional network model structure 
    training options, evaluatioon options, 
    data input functions. and GPU configuration
  preclas.py:
    preclassifier
  singleCam.py:
    download images from a single camera every 10 minutes, 
    manually kill the program when The time is long enough.

Folders:
  singleCamImg:
    Images downloaded from a single camera by singleCam.py
  tensorboard:
    stores the checkpoints of trained neural network
    please use command 'tensorboard --logdir tensorboard/model'
    to start the tendorboard web server and try 'localhost:6006' 
    in brower
    'localhost:6006' 
    
Files:
  x.npy:
    dataset inputs
  y.npy:
    dataset lables
    
