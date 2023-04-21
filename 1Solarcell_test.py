# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/1/26 16:56
# @Author : liumin
# @File : 1Solarcell_test.py

import math
import os
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from yolov5.models.yolo import Model
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords

warnings.simplefilter(action='ignore', category=FutureWarning)

torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

torch.set_default_tensor_type(torch.FloatTensor)

class SolarAlg(object):
    def __init__(self):
        super(SolarAlg, self).__init__()
        self.batch_size = 1 # 32
        self.num_workers = 1 # 8
        self.device = 'cuda:0'

        # weight_path = 'scripts/weights/df_11bb_danjing_bp_solarcell.pt'
        # weight_path = '/home/lmin/pythonCode/yolov5/runs/train/solarcell_960*480/weights/best_ema.pt'
        weight_path = 'best_ema22.pth' # 'best_ema.pt'
        model_path = 'models/yolov5x_solarcell_210.yaml' # '/home/lmin/pythonCode/yolov5/models/yolov5x_solarcell.yaml'
        if torch.cuda.is_available():
            model = Model(cfg=model_path)
            state_dictBA = torch.load(weight_path, map_location='cpu')['model']
            new_state_dictBA = OrderedDict()
            for k, v in state_dictBA.items():
                if k[:7] == 'module.':
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dictBA[name] = v
            model.load_state_dict(new_state_dictBA, False)
            # model.float().fuse().eval()
            model = model.float().eval()
            '''
            aaa = model.float()
            bbb = aaa.fuse()
            ccc = bbb.eval()
            '''
            self.model = model.to(self.device)
            # self.model.half()

        '''
            for m, n in model.named_parameters():
                print(m, n)
                
            for i, (m, n) in enumerate(aaa.named_parameters()):
                print(m, n)
                if i > 20:
                    break
        '''

        self.imgsize = (960, 480)
        self.iou_thres = 0.1
        self.names = ['handaipianyi', 'yiwu', 'quejiao', 'handaiqueshi', 'handaichaochuhuiliutiao']
        self.conf_thres = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.stride = 32
        self.auto = True


    def run(self, img):
        self.height, self.width, _ = img.shape

        # Convert
        # image1 = cv2.resize(img, self.imgsize[::-1])
        # cv2.imwrite('image1.jpg', image1)
        '''
        image = image1.astype(np.float32)
        # image = image.half()
        image /= 255.0
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image.copy()).to(self.device)
        '''

        # Padded resize
        image1 = letterbox(img, 960, stride=self.stride, auto=self.auto)[0]
        # Convert
        image1 = image1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image1 = np.ascontiguousarray(image1)
        image = torch.from_numpy(image1).to(self.device)
        image = image.float()
        # image = image.half()
        image /= 255  # 0 - 255 to 0.0 - 1.0
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim

        # torch.save(image, '/home/lmin/pythonCode/datasets/solarcell/image.pth')
        # image = torch.load('/home/lmin/pythonCode/datasets/solarcell/im.pth')
        with torch.no_grad():
            preds = self.model(image)[0]
        print(preds)
        # image11 = torch.load('/home/lmin/pythonCode/datasets/solarcell/im.pth')
        # image11 = image11.half()
        # preds11 = self.model(image11)[0]
        # print(preds11)

        # Apply NMS
        '''
        pred1 = torch.load('/home/lmin/pythonCode/datasets/solarcell/pred1.pth')
        pred2 = torch.load('/home/lmin/pythonCode/datasets/solarcell/pred2.pth')
        print('pred1', pred1)
        print('pred2', pred2)
        pred3 = non_max_suppression(pred1, 0.2, self.iou_thres)[0]
        print('pred3', pred3)

        det1 = torch.load('/home/lmin/pythonCode/datasets/solarcell/det1.pth')
        det2 = torch.load('/home/lmin/pythonCode/datasets/solarcell/det2.pth')
        print('det1', det1)
        print('det2', det2)
        # Rescale boxes from img_size to im0 size
        det1[:, :4] = scale_coords([960, 416], det1[:, :4], (924, 372, 3)).round()
        print('det3', det1)
        '''
        preds = non_max_suppression(preds, 0.2, self.iou_thres)[0]
        # preds = non_max_suppression_list(preds, self.conf_thres, self.iou_thres)[0]
        print(preds)
        print(img.shape)
        # print(image1.shape)
        print(image.shape)


        # cv2.rectangle(image1, (191, 207), (483, 426), (0, 255, 0), 1, 8)
        # cv2.imwrite('image2.jpg', image1)

        if preds is not None and len(preds):
            # Rescale boxes from img_size to im0 size
            # preds[:, :4] = scale_coords(image.shape[2:], preds[:, :4], img.shape).round()
            # images[idx].shape[1:], pred[:, :4], imgs[idx].shape
            preds[:, :4] = scale_coords(image.shape[2:], preds[:, :4], img.shape).round()
            print(preds)
            '''
            coords = preds[:, :4]
            height_ratio, width_ratio = img.shape[0] / image.shape[2], img.shape[1] / image.shape[3]
            print(height_ratio, width_ratio)
            coords[:, [0, 2]] *= width_ratio
            coords[:, [1, 3]] *= height_ratio
            preds[:, :4] = coords
            '''
            return preds
        return 1

if __name__ == '__main__':
    model1 = SolarAlg()

    # imgPath = "821000839239126273.jpg"
    # rgbImg = cv2.imread(imgPath)
    import time

    imgPath = "821009039267014366_6#17_6_17.jpg"
    rgbImg = cv2.imread(imgPath)

    since = time.time()
    rst = model1.run(rgbImg)
    time_elapsed = time.time() - since
    print("Time used:", time_elapsed)

    since = time.time()
    # rgbImg_show = drawResult(rgbImg,rst)
    # [[148.2473, 199.2666, 374.6540, 410.4592,   0.9560,   1.0000]]

    ## [[168.9411, 207.2595, 403.8045, 426.5775,   0.9348,   1.0000]]
    for rs in rst:
        cv2.rectangle(rgbImg, (int(rs[0]), int(rs[1])), (int(rs[2]), int(rs[3])), (0, 255, 0), 1, 8)
    time_elapsed = time.time() - since
    img_name = os.path.split(imgPath)[1]
    print("_drawResult used:", time_elapsed)
    cv2.imwrite('rgbImg_show_' + imgPath, rgbImg)

    # print(rst)
    print('Done!')