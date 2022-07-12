import torch
import math
import random

def gradients(x):
    h_x = x.size()[2]  # height
    w_x = x.size()[3]  # width
    # compute gradients ∂x, ∂y (-1, 1)
    x_grad_x = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
    x_grad_y = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
    x_grad_x = torch.cat((x_grad_x, x[:, :, :, 0:1] - x[:, :, :, w_x-1:w_x]), dim=3)
    x_grad_y = torch.cat((x_grad_y, x[:, :, 0:1, :] - x[:, :, h_x-1:h_x, :]), dim=2)

    return x_grad_x, x_grad_y


def truncated(grad, sigma):
    batchsz = grad.size()[0]
    # sum 3-channel to 1-channel
    temp = torch.sum(torch.abs(grad), dim=1).unsqueeze(1)
    # if the pixel less than sigma, set it to 0.
    zeros = torch.zeros((batchsz, 1, 1, 1), device="cpu")
    ones = torch.ones((batchsz, 1, 1, 1), device="cpu")
    # sigma = torch.zeros((batchsz, 1, 1, 1), device="cuda")
    # for i in range(batchsz):
    #     a = round(random.uniform(0.01, 1.2), 3)
    #     sigma[i, 0, 0, 0] = a
    temp = torch.where(temp < sigma, zeros, ones) #当满足条件就返回 zero,不满足返回ones 
    print(temp)
    gradients = torch.mul(grad, temp) #大于sigma梯度不变，小于sigma是压成0 大梯度时梯度不变，小梯度时为0
    print(gradients)
    return gradients
    

def gauss(ins, sigma):
    in_x, in_y = gradients(ins)
    # a = torch.sum(torch.pow(in_x, 2), dim=1).unsqueeze(1)
    # b = torch.sum(torch.pow(in_y, 2), dim=1).unsqueeze(1)
    a = torch.pow(in_x, 2)
    b = torch.pow(in_y, 2)
    a = -a / (2.0 * torch.pow(sigma, 2))
    b = -b / (2.0 * torch.pow(sigma, 2))
    w_x = torch.exp(a)
    w_y = torch.exp(b)
    return w_x, w_y


def randSigma(bz, h, w):
    sigma = torch.zeros((bz, 1, h, w), device="cuda")
    for i in range(bz):
        a = round(random.uniform(0.001, 0.1), 3)
        # a = 0.5
        sigma[i, 0, 0:h, 0:w] = a

    return sigma


def randSigma2(bz, h, w):
    sigma = torch.zeros((bz, 1, h, w), device="cuda")
    for i in range(bz):
        a = round(random.uniform(0.001, 1), 3)
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
if __name__ == "__main__":
    a = torch.randn((4, 3, 5, 5))
    print(a)
    b=truncated(a,1.5)
    print(b)
