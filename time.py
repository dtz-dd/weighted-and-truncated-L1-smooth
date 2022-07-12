"""
@file: test.py
@intro: this file is to test the effect of trained model.
@date: 2021/03/18 18:11:35
@author: tangling
@version: 1.0
"""

import time
import torch
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from net.net import RAE_1
from net.Unet import Unet
import os


def test():
    model = Unet()
    V = "v7.9"
    model.load_state_dict(torch.load("D:/labdata/truncated-and-weighted-L1-smooth/model/%s/_Unet.pth"%(V)))
    # model = model.to(torch.device('cuda'))
    model = model.to(torch.device('cpu'))
    
    # dirPath = 'D:/labdata/groundtruth/320X240'
    # dirPath = 'D:/labdata/groundtruth/640X480'
    dirPath = 'D:/labdata/groundtruth/1280X720'
    # fout = open("D:/labdata/truncated-and-weighted-L1-smooth/time/"+'320X240'+"time.csv",'w')
    # fout = open("D:/labdata/truncated-and-weighted-L1-smooth/time/"+'640X480'+"time.csv",'w')
    fout = open("D:/labdata/truncated-and-weighted-L1-smooth/time/"+'1280X720'+"time.csv",'w')
    # dirPath = 'D:/labdata/Filterings-based-on-Soft-Clustering-main/Filterings-based-on-Soft-Clustering-main/image/abstract'
    filenames = os.listdir(dirPath)
    for lam_value in range(2,3):
        print(lam_value)
        s_v = [ 0.2]
        for s in range(len(s_v)): #0.001-0.2
            for fn in filenames:
                if fn == '.DS_Store':
                    continue
                # fn = '18.tif'
                filename = os.path.join(dirPath, fn)
                # filename = "purp.jpg"
                # img = cv2.imread('F:/LabWork/contrast_results/test_v1.0/testSet/'+filename)
                img = cv2.imread(filename)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32)
                img = np.transpose(img, [2, 0, 1]) / 255.0
                img = torch.tensor(img)
                height = img.size()[1]
                width = img.size()[2]
                lam = torch.zeros((1, height, width), dtype=torch.float32)
                # lam_value = 2
                lam[:, 0: height, 0: width] = lam_value
                sigma = torch.zeros((1, height, width), dtype=torch.float32)
                # s_value = 0.15
                s_value = s_v[s]
                sigma[:, 0:height, 0:width] = s_value
                x = torch.cat((img, lam, sigma), dim=0)
                x = x.unsqueeze(0)
                
                # x = x.to(torch.device("cuda"))
                x = x.to(torch.device("cpu"))
                start = time.time()
                with torch.no_grad():
                    model.eval()
                    y = model(x)
                end = time.time()
                # writer = SummaryWriter()
                # writer.add_image("input", x[0, 0:3, ...])
                # writer.add_image("output", y[0])
                y = (y.cpu().numpy())
                y = np.uint8(np.minimum(np.maximum(y,0.0),1.0)*255.0)
                y = np.transpose(y[0], [1, 2, 0])
                # fout = open("D:/labdata/truncated-and-weighted-L1-smooth/time/"+'320X240'+"time.csv",'w')
                # fout.write("PSNR_"+str(lamd)+'_'+str(sigma)+'/n')
                

                # y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
                # out = "./results/%s/lam=%d&sigma=%1.2f"%(V,lam_value, s_value)
                # out = "./abstract_smooth/%s/lam=%d&sigma=%1.2f"%(V,lam_value, s_value)
                # if not os.path.exists(out):
                #     os.makedirs(out)
                # cv2.imwrite("%s/%s"%(out,fn), y)
                time1 = end-start
                print(time1)
                a = ''
                a += str(time1)
                fout.write(a + "\n")
    fout.flush()
    fout.close()

if __name__ == "__main__":
    test()
    # model = Unet()
    # model.load_state_dict(torch.load('./model/v7.3/_Unet.pth'))
    # torch.save(model.state_dict(), './model/v7.4/_Unet.pth', _use_new_zipfile_serialization=False)
