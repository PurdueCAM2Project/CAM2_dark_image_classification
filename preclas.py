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
            x=0
            for k in range(0,len(im[0][0])):
                x+=im[i,j,k]
            x/=3
            sum += pow((x - avg), 2)
    sum/=(len(im)*len(im[0]))
    sum=pow(sum,0.5)
    return sum


def preclassify(img_array):
    """

    :param img_array: input image array size = [num_of_img, x_size,y_size, color_channels]
    :return: pre-classified one-hot label
    """
    badlist=[]
    onehot_label=np.ndarray(shape=[len(img_array),2])
    for i in range(0,len(img_array)):
        avg=imgavg(img_array[i])
        std=imgstd(img_array[i])
        # print('avg=%d'%avg)
        # print('std=%d'%std)
        if avg<30 and std<10:
            onehot_label[i]=[0,1]
            badlist.append(i)
        else:
            onehot_label[i]=[1,0]


    return [onehot_label ,badlist]

def writeImg(imgarr, label):
    """

    :param imgarr: input image array size = [num_of_img, x_size,y_size, color_channels]
    :param label: pre-classified one-hot label
    :return: None
    """
    for i in range(0, len(imgarr)):
        im = Image.fromarray(imgarr[i])
        if label[i][0]==0:
            im.save('bad/%d.png'%i)
        else:
            im.save('good/%d.png'%i)
    return





def labelcorrection(reverse_list,onehot_label):
    """

    :param reverse_list: image index that has wrong label
    :param onehot_label: onehot-label to be corrected
    :return: corrected label
    """
    for i in reverse_list:
        onehot_label[i]=[1,1]-onehot_label[i]
    return onehot_label


def main():
    im=np.load('x.npy')
    label,bad=preclassify(im)
    writeImg(im, label)
    labelcorrection(rev_list, label)
    np.save(file='y.npy',arr=label)



rev_list = [] ## add manually checked mistake image index here
def correction():
    im=np.load('x.npy')
    label,bad=preclassify(im)
    label=labelcorrection(rev_list, label)
    np.save(file='y.npy',arr=label)

if __name__ == "__main__":
    correction()
    # main()
