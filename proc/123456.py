# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/4/29 18:07
# @Author : liumin
# @File : 123456.py

import os
import sys


def ppp():
    for name in ['handaipianyi', 'yiwu', 'quejiao', 'handaiqueshi', 'handaichaochuhuiliutiao']:
        os.system('cp -r /home/lmin/data/solarcell/51_solarcell_'+name+'/labels/train2017/* /home/lmin/data/solarcell/51_solarcell_'+name+'/labels/val2017/')
        os.system('cp -r /home/lmin/data/solarcell/51_solarcell_'+name+'/train2017/* /home/lmin/data/solarcell/51_solarcell_'+name+'/images/val2017/')


ppp()
print('finished!')