from scipy.io import loadmat
from IPython import embed
from PIL import Image
from tqdm import tqdm
import os
import cv2
import string
import numpy
import math
import json
import argparse


def t_split(txt):
    list1 = []
    for i in txt:
        c = i.split(' ')
        for t in c:
            tt = t.split('\n')
            for ttt in tt:
                if ttt != '':
                    list1.append(ttt)
    return list1


def gt_box(contours):
    list_bbox = []
    for j in range(contours.shape[2]):
        quad_points = numpy.zeros((4, 1, 2), dtype='float32')
        for k in range(4):
            quad_points[k][0][0] = contours[0][k][j]
            quad_points[k][0][1] = contours[1][k][j]
        list_bbox.append(quad_points)  # quad_points.astype(numpy.int32)
    return list_bbox


def crop_rect(list_bbox):
    list_rect = []

    for i in range(len(list_bbox)):
        temp = list_bbox[i].reshape(4, 2)
        temp1 = numpy.transpose(temp)
        x_min = math.floor(temp1[0].min())
        x_max = math.ceil(temp1[0].max())
        y_min = math.floor(temp1[1].min())
        y_max = math.ceil(temp1[1].max())

        quad_points = numpy.zeros((4, 1, 2), dtype='int32')
        quad_points[0][0] = [x_min, y_min]
        quad_points[1][0] = [x_max, y_min]
        quad_points[2][0] = [x_max, y_max]
        quad_points[3][0] = [x_min, y_max]
        list_rect.append(quad_points)
    return list_rect


def main(args):
    m = loadmat(args.gt_path)
    cwd = os.getcwd()
    # symbol = ['~','`','!','@','$','#','%','^','&','*','(',')','-','+','=','_',
    #           '|',',','[',']','\\',':',';','\'','"','.','?','<','>','/','{','}']
    symbol = string.punctuation
    count = 0
    for i in tqdm(range(m['imnames'][0].shape[0])):

        contours = m['wordBB'][0][i]  # note: cotours.shape=(2,4,len(labels))
        if len(contours.shape) == 2:
            contours = contours.reshape(2, 4, 1)
        im_name = m['imnames'][0][i][0]
        txt = m['txt'][0][i]

        labels = t_split(txt)

        BBox = gt_box(contours)  # the list of groundtruth boundingboxs,quadrilateral shape
        RectBox = crop_rect(BBox)  # the list of minimum circumscribed rectangle of the BBOXs
        count += contours.shape[2]
        path = os.path.join(args.syntext_path, im_name)
        im = cv2.imread(path)

        for j in range(len(labels)):
            x_min = max(RectBox[j][0][0][0], 0)
            x_max = min(RectBox[j][1][0][0], im.shape[1])
            y_min = max(RectBox[j][0][0][1], 0)
            y_max = min(RectBox[j][2][0][1], im.shape[0])
            # these 4 lines are used to ensure the box does not stretch out of the image

            im1 = im[y_min:y_max, x_min:x_max]
            label = labels[j]
            for idx in range(len(symbol)):
                label = label.replace(symbol[idx], '')

            new_name = im_name.split('/')[0] + '_' + im_name.split('/')[1].split('.')[0] + '_' + str(
                j) + '_' + label + '.jpg'
            im_new_path = os.path.join(args.out_path, 'syntxt_crop', im_name.split('/')[0])
            if not os.path.exists(im_new_path):
                os.mkdir(im_new_path)
            cv2.imwrite(os.path.join(im_new_path, new_name), im1)

            with open('./syntxt_crop.odgt', 'a') as f1:
                dict1 = {'im_path': str(im_new_path), 'im_name': str(new_name), 'label': str(labels[j])}
                js = json.dumps(dict1)
                f1.write(js + '\n')

        ''' 
        plot the bbox and rectangle of each word
        
        BBox1=[]
        for i in range(len(BBox)):
            BBox1.append(BBox[i].astype(numpy.int32))

        im1=cv2.polylines(im,RectBox,isClosed=True,color=(255,255,255),thickness=1,lineType=8)
        im2=cv2.polylines(im,BBox1,isClosed=True,color=(255,255,255),thickness=1,lineType=8)
        '''

        '''
        wordBoundingBox is based on four points (x1,y1),(x2,y2),(x3,y3),(x4,y4)
        each picture has n BOXes,which is same as the numbers of the labels.
        (the labels could be splited from the txt)
        the labels and the wordBBs are of one-to-one correspondence
        
        eg.
        m['wordBB'][0][12]=
        array([[[144.11255, 280.32397, 425.85638, 508.72253],                    ## x1  
                [221.60611, 421.7674 , 489.43262, 580.4785 ],                    ## x2
                [210.23676, 421.62714, 488.78485, 579.8429 ],                    ## x3      k
                [132.7432 , 280.18372, 425.20862, 508.0869 ]],                   ## x4      |
                                                                                            |
               [[203.68845, 384.30234, 359.90906, 360.9495 ],                    ## y1      |
                [223.97514, 385.26584, 360.70734, 361.85043],                    ## y2      *
                [267.40524, 405.8596 , 412.29712, 412.47342],                    ## y3
                [247.11855, 404.8961 , 411.49884, 411.57248]]], dtype=float32)   ## y4
                ##BOX1##    ##BOX2##    ##BOX3##    ##BOX4##
                                j----->
        m['imnames'][0][12]=
        array(['8/ballet_106_109.jpg'], dtype='<U20')
        
        m['txt'][0][12]=
        array(['the         ', '[Description', 'V8 V12      '], dtype='<U12')
        
        t_split(m['txt'][0][12])=
        ['the', '[Description', 'V8', 'V12']
        '''
    print(count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert images')
    parser.add_argument('--gt_path', default='./SynthText/gt.mat', type=str, help='')
    parser.add_argument('--syntxt_path', default='./SynthText', type=str, help='')
    parser.add_argument('--out_path', default='./', type=str, help='')
    args = parser.parse_args()
    main(args)
