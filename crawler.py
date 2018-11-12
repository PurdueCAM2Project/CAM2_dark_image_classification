
import csv
import urllib.request

import eventlet
import os
import numpy as np
from PIL import Image

#create a new directory to store downloadede image
if not os.path.exists('imageall1'):
    os.makedirs('imageall1')
dataset=[]

## csvfile format: ip,port, image_path
with open('mysqlout/bbb.csv','r') as csvfile:
    reader =csv.reader(csvfile)
    i=0
    for url in reader:
        path=('imageall1/' + str(i)+ ".jpg")
        urlstr='http://'+url[0]+':'+url[1]
        if(url[2]!='NULL'):
            urlstr=urlstr+url[2]
        print(urlstr)
        # print(i)
        try:
            urllib.request.urlretrieve(url=urlstr, filename=path)
            im = Image.open(path)
            (x, y) = im.size
            ##reshape image
            x_s = 40
            y_s = 30
            out = im.resize((x_s, y_s), Image.ANTIALIAS)
            im2 = np.array(out)
            ## fix the channels of images
            if (len(im2.shape) < 3):
                imx = np.ndarray([im2.shape[0], im2.shape[1], 3])
                for x in range(im2.shape[0]):
                    for y in range(im2.shape[1]):
                          imx[x][y][0] = im2[x][y]
                          imx[x][y][1] = im2[x][y]
                          imx[x][y][2] = im2[x][y]
            else:
                imx = im2
            print(imx.shape)
            dataset.append(imx)
            if (i % 50 == 0):
                np.save('./nightnpy/dataset%d.npy' % i, dataset)
                arr = np.load('./nightnpy/dataset%d.npy' % i, allow_pickle=True)
                print(arr.shape)
                # dataset.clear()
                # dataset=[]
                print(i)
            i += 1
        except Exception as e:
            print(e)

