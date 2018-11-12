import numpy as np
from PIL import Image

def imgavg(im):
    sum=0
    for i in range (0,len(im)):
        for j in range(0,len(im[0])):
            for k in range(0,len(im[0][0])):
                sum += im[i,j,k]
    sum/=(len(im)*len(im[0])*len(im[0][0]))
    return sum


def imgstd(im):
    avg=imgavg(im)
    sum=0;
    for i in range (0,len(im)):
        for j in range(0,len(im[0])):
            for k in range(0,len(im[0][0])):
                sum+=pow((im[i,j,k]-avg),2)
    sum/=(len(im)*len(im[0])*len(im[0][0]))
    sum=pow(sum,0.5)
    return sum

def preclassify(img_array):
    onehot_label=np.ndarray(shape=[len(img_array),2])
    for i in range(0,len(img_array)):
        avg=imgavg(img_array[i])
        std=imgstd(img_array[i])
        if(avg<50 or std<30):
            onehot_label[i]=[0,1]
        else:
            onehot_label[i]=[1,0]
    return onehot_label


def main():
    im=np.load('x.npy')
    label=preclassify(im)
    # print(label)
    np.save(file='y.npy',arr=label)
    print(im.shape)
    print(label.shape)




if __name__ == "__main__":
    main()