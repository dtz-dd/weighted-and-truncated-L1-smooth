"""
@file: train.py
@intro: this file is to train the model
@date: 2021/03/10 15:16:28
@author: tangling
@version: 1.0
"""


import torch
# from net.net import RAE, Unet, RAE_1
from net.Unet import Unet, Unet3
from lossFunction.loss import WLoss
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
from tool.util import gauss, gradients, trancated
import random


def loadData(path, batch_size):
    """
    load dataset
    :param path: the path of dataset
    :param batch_size: 
    :return:
    """
    train_set = datasets.ImageFolder(path, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]))
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    return train_set

def randSigma(bz, h, w):
    sigma = torch.zeros((bz, 1, h, w), device="cuda")
    for i in range(bz):
        a = round(random.uniform(0.1, 0.2), 3)
        # a = 0.5
        sigma[i, 0, 0:h, 0:w] = a

    return sigma


def randLambda(bz, h, w):
    lam = torch.zeros((bz, 1, h, w), device="cuda")
    for i in range(bz):
        a = round(random.uniform(0, 5), 2)
        # a = 0.01
        lam[i, 0, 0:h, 0:w] = a

    return lam


def train():
    trainSet = loadData('./pascal_train_set', 32)
    device = torch.device('cuda')
    model = Unet()
    model.load_state_dict(torch.load("./model/v7.5/_Unet.pth"))
    criterion  = WLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model = model.to(device)
    # 初始化可视化工具
    writer = SummaryWriter()
    x, _ = iter(trainSet).next()
    # 绘制网络结构
    # input_data = torch.randn((1, 5, 224, 224))
    # writer.add_graph(model=model, input_to_model=(input_data.cuda(), ))
    height = x.size()[2]
    width = x.size()[3]

    for epoch in range(500):
        running_loss = 0
        for bx, (img, _) in enumerate(trainSet):
            img = img.to(device)
            bz = img.size()[0]
            sigma = randSigma(bz, height, width)
            lam = randLambda(bz, height, width)
            x = torch.cat((img, lam, sigma), dim=1)

            y = model(x)
            loss = criterion(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        m_loss = running_loss / len(trainSet)
        print(epoch, 'train_loss:', m_loss)
        print(epoch, 'lam:', lam[:, 0, 0, 0])
        print(epoch, 'sigma:', sigma[:, 0, 0, 0])
        writer.add_scalar('train_loss', m_loss, epoch)
        writer.add_images(tag='train_input', img_tensor=img, global_step=epoch)
        writer.add_images(tag='train_output', img_tensor=y, global_step=epoch)

        if epoch and epoch % 100 == 0:
            torch.save(model.state_dict(), "./model/v7.5.2/{}_Unet.pth".format(epoch), _use_new_zipfile_serialization=False)   

    torch.save(model.state_dict(), "./model/v7.5.2/_Unet.pth", _use_new_zipfile_serialization=False)   


if __name__ == "__main__":
    train()

