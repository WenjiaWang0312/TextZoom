# This is a super-resolution dataset containing paired LR-HR scene text images.

## TextZoom Dataset (allocated by size): 

Paper: [arxiv](https://arxiv.org/abs/2005.03341)

Data: [Badiu NetDisk](https://pan.baidu.com/s/1PYdNqo0GIeamkYHXJmRlDw). password: **kybq**

The LR images in TextZoom is much more challenging than synthetic LR images(BICUBIC).

<img src="syn_real.jpg" width=80% />

We allocate our dataset into 3 part following difficulty: easy, medium and hard subset. The misalignment and ambiguity increases as the difficulty increases.

<img src="easy_medium_hard.jpg" width=80% />

For each pair of LR-HR images, we provide the annotation of the case sensitive character string (including punctuation), the type of the bounding box, and the original focal lengths.

## Other data

- Cropped text images from SR_RAW (allocated by original images): [BaiduNet Disk](https://pan.baidu.com/s/1deWqGQTbiITrayFNrrJg-w).  password: **ykbq**

- Cropped text images from RealSR (allocated by original images): [BaiduNet Disk](https://pan.baidu.com/s/1gjwQ05THh-MJv3oChvm3FA).  password: **f615**

- Annotation of SR_RAW (bounding boxs and word labels): [Baidu NetDisk](https://pan.baidu.com/s/1OQpiItFTiYHhZyhbg1ASWg). password: **kmme**

<img src="sr_raw.jpg" width=80% />

- Annotation of RealSR (bounding boxs and word labels): [Baidu NetDisk](https://pan.baidu.com/s/19-_jnlxJhWrUs_2n9JUsiw). password: **i52c**

<img src="real_sr.jpg" width=50% />

```
    architecture of json: (sr_raw.json and real_sr.json have the same arch)

    'position' is the bounding box,

    'rawFileName' is the original image name, you need to download the RealSR dataset.

    'words' is the word label.

    'type' means the direction of bounding box, 'td' means top down, 'vn' means negative vertical (counterclockwise 90 degrees), 
    'vp' means positive vertical (clockwise 90 degrees), 'h' means horizontal.

    
    with open('real_sr.json') as f:
        d=json.load(f)
    d['0']=
    {'channal': '3',
     'height':  '2300',
     'id':      'cbe0e4cba6ba6cd42d8ed4779087214a',
     'polygons': {'wordRect': 
                 [{'line-type': 'straight',
                    'position': [{'x': '247.94625', 'y': '186.31634'},
                     {'x': '99.29263', 'y': '186.60167'},
                     {'x': '99.29263', 'y': '165.77304'},
                     {'x': '247.94625', 'y': '166.34369'}],
                    'type': 'td',
                    'valid': 'true',
                    **'words': 'QU04029757'**},
                   {'line-type': 'straight',
                    'position': [{'x': '63.18353', 'y': '703.61181'},
                     {'x': '61.66713', 'y': '542.87290'},
                     {'x': '127.88347', 'y': '540.85103'},
                     {'x': '130.41081', 'y': '702.60087'}],
                    'type': 'vn',
                    'valid': 'true',
                    'words': '100'},
                   ...
                   ]},
     'rawFilePath':   'test',
     'rawFilename':   'Canon_046_HR.png',
     'result_version': '1.0',
     'rotate':    '0',
     'valid':     'true',
     'width':     '2500',
     'wordRect-validity': 'true'}
```
