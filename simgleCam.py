import os
import urllib
import time
import sched
from PIL import Image
import numpy as np





dataset=[]
def downImg(url, path):
    try:
        urllib.request.urlretrieve(url=url, filename=path)
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
        dataset.append(imx)

    except Exception as e:
        print(e)

    return dataset

if not os.path.exists('singleCamImg'):
    os.makedirs('singleCamImg')
exampleurl = 'http://207.251.86.238/cctv202.jpg'
outpath = 'singleCamImg/'
timewait = 1200 ## take image per 20 minutes
def main():
    schedule = sched.scheduler(time.time, time.sleep)
    i=0
    while(1):
        path = outpath + time.strftime("%Y%m%d-%H-%M-%S.jpg", time.localtime(time.time()))
        print(path)
        schedule.enter(timewait, 0, downImg, (exampleurl, path))
        schedule.run()
        i+=1
        if(i%10==0):
            np.save('./nightnpy/single_img_x.npy' , dataset)
            arr = np.load('./nightnpy/single_img_x.npy' , allow_pickle=True)
            print(arr.shape)


if __name__ == "__main__":
    main()
