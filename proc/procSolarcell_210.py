# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/1/3 16:07
# @Author : liumin
# @File : procSolarcell_210.py

import os
import random
import sys
import cv2
import re
import shutil
from glob2 import glob

defect_cls = 'solarcell_210'


def get_file_path(root_path, file_list):
    #获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        #获取目录或者文件的路径
        dir_file_path = os.path.join(root_path,dir_file)
        #判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            #递归获取所有文件和目录的路径
             get_file_path(dir_file_path, file_list)
        else:
            if dir_file_path.endswith('.jpg'):
                file_list.append(dir_file_path)

repeat_list = []
def moveValidData():
    root_path = '/home/lmin/data/solarcell/' + defect_cls
    img_path = os.path.join(root_path, 'images/train2017')
    xml_path = os.path.join(root_path, 'annotations/train2017')
    org_path = os.path.join(root_path, 'org')
    imglist = []
    get_file_path(org_path, imglist)
    for imgPath in imglist:
        xmlPath = imgPath.replace('.jpg', '.xml')
        imgname = os.path.basename(imgPath)
        xmlname = imgname.replace('.jpg', '.xml')
        if os.path.exists(imgPath) and os.path.exists(xmlPath):
            if os.path.exists(os.path.join(img_path, imgname)) or os.path.exists(os.path.join(xml_path, xmlname)):
                repeat_list.append(imgname)
                continue
            print(imgPath, xmlPath)
            shutil.copyfile(imgPath, os.path.join(img_path, imgname))
            shutil.copyfile(xmlPath, os.path.join(xml_path, xmlname))

def produceImgAndLabelsList():
    root_path = '/home/lmin/data/solarcell/'+defect_cls
    seg_txt = open(root_path + '/img_list.txt', 'a')
    imglist = glob(root_path + "/images/train2017/*.jpg")
    xml_dir = root_path+'/annotations/train2017/'
    xmlpaths = glob(os.path.join(xml_dir, '*.xml'))
    for i, imgPath in enumerate(imglist):
        print(i, imgPath)
        imgname = os.path.basename(imgPath)
        spp = imgname.strip().split(' ')
        if len(spp) >1:
            continue
        xmlname = imgname.replace('.jpg', '.xml')
        if os.path.join(xml_dir, xmlname) in xmlpaths:
            seg_txt.write(imgname+' '+imgname.replace('jpg','xml') + '\n')
    seg_txt.close()


pattens = ['name', 'xmin', 'ymin', 'xmax', 'ymax']

def get_annotations(xml_path):
    bbox = []
    with open(xml_path, 'r') as f:
        text = f.read().replace('\n', 'return')
        p1 = re.compile(r'(?<=<object>)(.*?)(?=</object>)')
        result = p1.findall(text)
        for obj in result:
            tmp = []
            for patten in pattens:
                p = re.compile(r'(?<=<{}>)(.*?)(?=</{}>)'.format(patten, patten))
                if patten == 'name':
                    tmp.append(p.findall(obj)[0])
                else:
                    tmp.append(int(float(p.findall(obj)[0])))
            bbox.append(tmp)
    return bbox

# lbls = {'handaipianyi': 0, 'yiwu': 1, 'quejiao': 2, 'handailianjie':3, 'handaiqueshi':4, 'handaichaochuhuiliutiao':5 }
lbls = {'handaipianyi': 0, 'yiwu': 1, 'quejiao': 2, 'handaiqueshi':3, 'handaichaochuhuiliutiao':4 }

def transVoc2Yolo():
    lbl_num = [0] * len(lbls)
    hw_ratios = [[0] for _ in lbls]

    rootpath = '/home/lmin/data/solarcell/' + defect_cls
    file = open(rootpath + '/img_list.txt')
    lines = file.readlines()  # 读取全部内容
    for line in lines:
        imgpath, xmlpath = line.strip().split(' ')
        print(imgpath, xmlpath)
        image = cv2.imread(os.path.join(rootpath, 'images/train2017', imgpath))
        gheight, gwidth, _ = image.shape
        bbox = get_annotations(os.path.join(rootpath, 'annotations/train2017', xmlpath))

        seg_txt = open(rootpath + '/labels/train2017/' + imgpath[:-4] + '.txt', 'a')
        for bb in bbox:
            if bb[0] not in lbls:
                continue
            cls = str(lbls[bb[0]])
            bb[1], bb[2], bb[3], bb[4] = max(bb[1],0), max(bb[2],0), min(bb[3],gwidth-1), min(bb[4],gheight-1)
            lbl_num[int(cls)] = lbl_num[int(cls)] + 1
            x_center = str((bb[1] + bb[3]) * 0.5 / gwidth)
            y_center = str((bb[2] + bb[4]) * 0.5 / gheight)
            width = str((bb[3] - bb[1]) * 1.0 / gwidth)
            height = str((bb[4] - bb[2]) * 1.0 / gheight)
            seg_txt.write(cls + ' ' + x_center + ' ' + y_center + ' ' + width + ' ' + height + '\n')

            hw_ratios[int(cls)].append(max((bb[3] - bb[1]) / (bb[4] - bb[2]), (bb[4] - bb[2]) / (bb[3] - bb[1])))

        seg_txt.close()

    file.close()

    print(lbl_num)
    print('------')
    for hw_ratio in hw_ratios:
        print(min(hw_ratio[1:]), max(hw_ratio[1:]))


def splitTrainVal(p=0.2):
    root_path = '/home/lmin/data/solarcell/' + defect_cls
    img_path = os.path.join(root_path, 'images/train2017')
    xml_path = os.path.join(root_path, 'labels/train2017')

    val_img_path = os.path.join(root_path, 'images/val2017')
    val_xml_path = os.path.join(root_path, 'labels/val2017')
    imglist = os.listdir(img_path)
    print('total:', len(imglist))
    for imgname in imglist:
        imgpath = os.path.join(img_path, imgname)
        if random.random() < p:
            xmlname = imgname.replace('.jpg', '.txt')
            # shutil.copyfile(imgpath, os.path.join(val_img_path, imgname))
            # shutil.copyfile(os.path.join(xml_path, xmlname), os.path.join(val_xml_path, xmlname))
            print(imgpath, os.path.join(val_img_path, imgname))
            print(os.path.join(xml_path, xmlname), os.path.join(val_xml_path, xmlname))
            shutil.move(imgpath, os.path.join(val_img_path, imgname))
            shutil.move(os.path.join(xml_path, xmlname), os.path.join(val_xml_path, xmlname))


## [2395, 3814, 513, 127, 7112, 4726]
## [2395, 3821, 525, 127, 7113, 4726]
moveValidData()
produceImgAndLabelsList()
transVoc2Yolo()
# splitTrainVal()

print('repeat_list', repeat_list)