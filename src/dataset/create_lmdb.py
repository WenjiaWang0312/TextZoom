import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import scipy.io as sio
from scipy.io import loadmat
from tqdm import tqdm
import re, time
import random
import six
from IPython import embed


def rand_crop(im):
    w, h = im.size
    scale = 0.95
    p1 = (random.uniform(0, w * (1 - scale)), random.uniform(0, h * (1 - scale)))
    p2 = (p1[0] + scale * w, p1[1] + scale * h)
    return im.crop(p1 + p2)


def buf2PIL(txn, key, type='RGB', degree=0):
    imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    im = Image.open(buf).convert(type)
    if degree is not 0:
        im = im.rotate(angle=degree, resample=Image.BICUBIC)
    return im


def PIL2buf(im):
    name = str(int(time.time())) + '.jpg'
    im.save(os.path.join('./', name))
    with open(os.path.join('./', name), 'rb') as f:
        imageBin = f.read()
    os.remove(os.path.join('./', name))
    return imageBin


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) is not bytes:
                k = k.encode()
            txn.put(k, v)


def _is_difficult(word):
    assert isinstance(word, str)
    return not re.match('^[\w]+$', word)


def createDataset_detection(outputPath, imagePathList, box_x_list, box_y_list,
                            labelList, region_mask_list, pixel_mask_list):
    """
  Create LMDB dataset for CRNN training.
  ARGS:
      outputPath      : LMDB output path
      imagePathList   : list of image path
      labelList       : list of corresponding groundtruth texts
      box_x_list      : coordinate x of gt_boxed
      box_y_list      : coordinate y of gt_boxed
      region_mask_list: region_mask images
      pixel_mask_list : pixel_mask images
  """
    assert (len(imagePathList) == len(box_x_list) == len(box_y_list))
    nSamples = len(imagePathList)
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        box_x = box_x_list[i]
        box_y = box_y_list[i]
        if len(box_x) == 0:
            continue
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        imageKey = 'image-%09d' % cnt
        cache[imageKey] = imageBin
        box_x_Key = 'boxes_x-%09d' % cnt
        box_y_Key = 'boxes_y-%09d' % cnt
        cache[box_x_Key] = box_x.encode()
        cache[box_y_Key] = box_y.encode()

        if labelList:
            labelKey = 'label-%09d' % cnt
            cache[labelKey] = labelList[i].encode()
        if region_mask_list:
            region_mask_Key = 'region_mask-%09d' % cnt
            cache[region_mask_Key] = open(region_mask_list[i], 'rb').read()
        if pixel_mask_list:
            pixel_mask_Key = 'pixel_mask-%09d' % cnt
            cache[pixel_mask_Key] = open(pixel_mask_list[i], 'rb').read()
        # embed()
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def createDataset_inpainting(outputPath, imagePathList, labelList):
    """
  Create LMDB dataset for CRNN training.
  ARGS:
      outputPath      : LMDB output path
      imagePathList   : list of image path
      labelList       : list of corresponding groundtruth texts
  """
    assert (len(imagePathList) == len(box_x_list) == len(box_y_list))
    nSamples = len(imagePathList)
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        box_x = box_x_list[i]
        box_y = box_y_list[i]
        if len(box_x) == 0:
            continue
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        imageKey = 'image-%09d' % cnt
        cache[imageKey] = imageBin
        box_x_Key = 'boxes_x-%09d' % cnt
        box_y_Key = 'boxes_y-%09d' % cnt
        cache[box_x_Key] = box_x.encode()
        cache[box_y_Key] = box_y.encode()

        if labelList:
            labelKey = 'label-%09d' % cnt
            cache[labelKey] = labelList[i].encode()
        if region_mask_list:
            region_mask_Key = 'region_mask-%09d' % cnt
            cache[region_mask_Key] = open(region_mask_list[i], 'rb').read()
        if pixel_mask_list:
            pixel_mask_Key = 'pixel_mask-%09d' % cnt
            cache[pixel_mask_Key] = open(pixel_mask_list[i], 'rb').read()
        # embed()
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
  Create LMDB dataset for CRNN training.
  ARGS:
      outputPath    : LMDB output path
      imagePathList : list of image path
      labelList     : list of corresponding groundtruth texts
      lexiconList   : (optional) list of lexicon lists
      checkValid    : if true, check the validity of every image
  """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if len(label) == 0:
            continue
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin

        cache[labelKey] = label.encode()
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        # embed()
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def create_800k():
    json_path = '/home/wwj/0_LAB/dataset/SynthText/syntxt_crop.odgt'
    lmdb_output_path = '/home/wwj/0_LAB/dataset/LMDB/syn800k_HR2'
    if not os.path.exists(lmdb_output_path):
        os.mkdir(lmdb_output_path)
    with open(json_path, 'r') as f:
        data = f.readlines()[:]

    data = [json.loads(i) for i in data]
    image_names = []
    image_paths = []
    image_labels = []
    portion = 1
    w_list = []
    h_list = []
    for i in tqdm(data):
        image_path = i['im_path']
        image_name = i['im_name']
        image_label = i['label']

        try:
            temp = Image.open(os.path.join(image_path, image_name))
            w = temp.size[0]
            h = temp.size[1]

            if h >= 64 and w >= 256:
                image_labels.append(image_label)
                image_names.append(image_name)
                image_paths.append(os.path.join(image_path, image_name))

        except OSError:
            pass
    print('there are all %d images' % len(image_paths))

    createDataset(lmdb_output_path, image_paths, image_labels)


def create_mat():
    image_paths = []
    image_labels = []

    lmdb_output_path = '../../dataset/LMDB/iiit5k_train'
    if not os.path.exists(lmdb_output_path):
        os.mkdir(lmdb_output_path)
    root = '../../dataset/IIIT5K'
    train_gt = loadmat(os.path.join(root, 'traindata.mat'))
    length = train_gt['traindata'][0].__len__()
    for i in tqdm(range(length)):
        im_path = train_gt['traindata'][0][i][0][0]
        im_gt = train_gt['traindata'][0][i][1][0]
        # lexi_50 = train_gt['traindata'][0][i][2][0]  #len==50
        # lexi_1k = train_gt['traindata'][0][i][3][0]   #len==1k

        try:
            image_path = os.path.join(root, im_path)
            image_label = im_gt
            temp = Image.open(image_path)
            # w = temp.size[0]
            # h = temp.size[1]
            # portion1 = w / float(h)
            # w_lsit.append(w)
            # h_list.append(h)
            image_labels.append(image_label)
            image_paths.append(image_path)
        except OSError:
            pass
    # embed()
    print('there are all %d images' % len(image_paths))
    createDataset_detection(lmdb_output_path, image_paths, image_labels)


def create_total_text():
    image_paths = []
    image_labels = []
    box_x_list = []
    box_y_list = []
    region_mask_list = []
    pixel_mask_list = []
    type = 'Test'
    lmdb_output_path = '../../dataset/LMDB/total_text_' + type

    if not os.path.exists(lmdb_output_path):
        os.mkdir(lmdb_output_path)
    root = '../../dataset/total_text'
    gt_root = os.path.join(root, 'Groundtruth', 'Polygon', type)
    region_mask_dir = os.path.join(root, 'Text_Region_Mask', type)
    pixel_mask_dir = os.path.join(root, 'groundtruth_pixel', type)
    im_dir = os.path.join(root, 'Images', type)
    im_names = os.listdir(im_dir)

    for name in tqdm(im_names):
        try:
            image_path = os.path.join(im_dir, name)
            temp = Image.open(image_path)

            region_mask_name = name.split('.')[0] + '.png'
            pixel_mask_name = name
            region_mask_path = os.path.join(region_mask_dir, region_mask_name)
            pixel_mask_path = os.path.join(pixel_mask_dir, pixel_mask_name)

            mat_name = ('poly_' if type == 'Test' else '') + 'gt_' + name.split('.')[0] + '.mat'
            mat_path = os.path.join(gt_root, mat_name)
            gt = loadmat(mat_path)
            gt_mat = gt[('poly' if type == 'Test' else '') + 'gt']
            sub = len(gt_mat)
            box_xs = []
            box_ys = []
            labels = []
            for i in range(sub):
                box_x = str(list(gt_mat[i][1][0])).replace('[', '').replace(']', '').replace(' ', '')
                box_y = str(list(gt_mat[i][3][0])).replace('[', '').replace(']', '').replace(' ', '')
                label = gt_mat[i][4][0]
                box_xs.append(box_x)
                box_ys.append(box_y)
                labels.append(label)
            box_x_str = ' '.join(box_xs)
            box_y_str = ' '.join(box_ys)
            label_str = ' '.join(labels)

            box_x_list.append(box_x_str)
            box_y_list.append(box_y_str)
            region_mask_list.append(region_mask_path)
            pixel_mask_list.append(pixel_mask_path)
            image_labels.append(label_str)
            image_paths.append(image_path)
        except OSError:
            embed()
    embed()
    print('there are all %d images' % len(image_paths))

    createDataset_detection(lmdb_output_path, image_paths, box_x_list, box_y_list,
                            image_labels, region_mask_list, pixel_mask_list)


def create_90k():
    image_paths = []
    image_labels = []
    lmdb_output_path = '/home/wwj/0_LAB/dataset/LMDB/syn90k_hr'
    root = '/home/wwj/0_LAB/dataset/mjsynth/mnt/ramdisk/max/90kDICT32px'
    Dirs = os.listdir(root)
    w_lsit = []
    h_list = []
    for i in Dirs:
        if '.' in i:
            Dirs.remove(i)
    print('there are all %d directories' % len(Dirs))
    for i in tqdm(Dirs):
        dirs = os.listdir(os.path.join(root, i))
        for dir in dirs:
            images = os.listdir(os.path.join(root, i, dir))
            for image in images:
                try:
                    image_path = os.path.join(root, i, dir, image)
                    image_label = image.split('_')[1]
                    temp = Image.open(image_path)
                    w = temp.size[0]
                    h = temp.size[1]
                    portion1 = w / float(h)
                    w_lsit.append(w)
                    h_list.append(h)
                    # dump the foo fat or thin images
                    if w >= 100 and h >= 31:
                        image_labels.append(image_label)
                        image_paths.append(os.path.join(image_path))

                except OSError:
                    pass
    # embed()
    print('there are all %d images' % len(image_paths))
    createDataset(lmdb_output_path, image_paths, image_labels)


def create_ic():
    json_path = '/home/wwj/0_LAB/dataset/ic.odgt'
    lmdb_output_path_13train = '/home/wwj/0_LAB/dataset/LMDB/ic13_train'
    lmdb_output_path_13test = '/home/wwj/0_LAB/dataset/LMDB/ic13_test'
    lmdb_output_path_15train = '/home/wwj/0_LAB/dataset/LMDB/ic15_train'
    lmdb_output_path_15test = '/home/wwj/0_LAB/dataset/LMDB/ic15_test'

    image_paths_13train = []
    image_labels_13train = []
    image_paths_13test = []
    image_labels_13test = []
    image_paths_15train = []
    image_labels_15train = []
    image_paths_15test = []
    image_labels_15test = []

    with open(json_path, 'r') as f:
        data = f.readlines()
    data = [json.loads(i) for i in data]

    for i in data:
        image_name = i['img_path']
        image_name = os.path.join('/home/wwj/0_LAB/SRGAN/ic13', image_name.split('/')[-1])
        image_label = i['img_gt']
        data_set = i['dataset']
        data_type = i['type']
        if os.path.exists(image_name):
            try:
                temp = Image.open(os.path.join(image_name))
                w = temp.size[0]
                h = temp.size[1]
                portion1 = w / h

                if data_set == 'IC13' and data_type == 'test':
                    image_labels_13test.append(image_label)
                    image_paths_13test.append(image_name)
                elif data_set == 'IC13' and data_type == 'train':
                    image_labels_13train.append(image_label)
                    image_paths_13train.append(image_name)
                elif data_set == 'IC15' and data_type == 'test':
                    image_labels_15test.append(image_label)
                    image_paths_15test.append(image_name)
                elif data_set == 'IC13' and data_type == 'test':
                    image_labels_13test.append(image_label)
                    image_paths_13test.append(image_name)
            except OSError:
                pass

    createDataset(lmdb_output_path_13train, image_paths_13train, image_labels_13train)
    createDataset(lmdb_output_path_13test, image_paths_13test, image_labels_13test)
    createDataset(lmdb_output_path_15train, image_paths_15train, image_labels_15train)
    createDataset(lmdb_output_path_13test, image_paths_13test, image_labels_13test)


def create_txt():
    root = '/home/wwj/0_LAB/dataset/STR/SVT-Perspective'
    image_labels = []
    image_paths = []
    lmdb_output_path = '/home/wwj/0_LAB/dataset/LMDB/svtp-645'
    with open(os.path.join(root, 'gt.txt')) as f:
        gts = f.readlines()
    for gt in gts:
        gt = gt.split(' ')
        im_name = gt[0]
        label = gt[1].replace('\r\n', '')
        image_labels.append(label)
        image_paths.append(os.path.join(root, im_name))

    createDataset(lmdb_output_path, image_paths, image_labels)


def create_from_lmdb():
    root = '/mnt/lustre/wangwenjia/wwj_space/dataset/lmdb/str/syn800k_HR2'
    out_path = '/mnt/lustre/wangwenjia/wwj_space/dataset/lmdb/str/syn800k_HR_crop'
    env = lmdb.open(root, map_size=1099511627776)
    env_out = lmdb.open(out_path, map_size=1099511627776)
    cache_out = {}
    txn = env.begin()
    num_samples = int(txn.get(b'num-samples'))
    for cnt in range(num_samples):
        imageKey = b'image-%09d' % (cnt + 1)
        image_HR_Key = 'image_HR-%09d' % (cnt + 1)
        image_lr_Key = 'image_lr-%09d' % (cnt + 1)
        image = buf2PIL(txn, imageKey)
        out_image = rand_crop(image)
        labelKey = b'label-%09d' % (cnt + 1)
        label = txn.get(labelKey)
        # embed()
        cache_out[image_HR_Key] = PIL2buf(image)
        cache_out[image_lr_Key] = PIL2buf(out_image)
        cache_out[labelKey] = label
        if cnt % 1000 == 0:
            writeCache(env_out, cache_out)
            cache_out = {}
            print('Written %d / %d' % (cnt, num_samples))

    cache_out['num-samples'] = str(num_samples).encode()
    writeCache(env_out, cache_out)
    print('Created dataset with %d samples' % num_samples)


if __name__ == "__main__":
    start_time = time.time()
    create_from_lmdb()
    end_time = time.time()
    print('cost %d seconds' % (end_time - start_time))
    #
    # lmdb_output_path = '/home/wwj/0_LAB/dataset/'
    # # gt_file = os.path.join(data_dir, 'gt.txt')
    # # image_dir = data_dir
    # # with open(gt_file, 'r') as f:
    # #   lines = [line.strip('\n') for line in f.readlines()]
    # lines=os.listdir(data_dir)
    #
    # imagePathList, labelList = [], []
    # for i, line in enumerate(lines):
    #   # splits = line.split(' ')
    #   # image_name = splits[0]
    #   # gt_text = splits[1]
    #   image_name=line
    #   gt_text=line.split('_')[0]
    #   #print(image_name, gt_text)
    #   imagePathList.append(os.path.join(data_dir, image_name))
    #   labelList.append(gt_text)
    #
    # embed()
    # createDataset(lmdb_output_path, imagePathList, labelList)
