# coding=utf8
# 你配不上自己的野心，也就辜负了先前的苦难
# 整段注释Control+/
import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms

#debug1:x需要再次归一化，以确保高斯核的和为1。
#debug2:imgs.squeeze方法应该指定要squeeze的维度，通常是批次维度，这里假设是第一个维度（索引为0）。
#debug3:with torch.no_grad():
class GaussianBlur:
    """blur a single image """
    def __init__(self,kernel_size):
        radius=kernel_size//2
        kernel_size=radius*2+1
        #kernel_size 的调整是为了确保高斯模糊操作的核大小是奇数，这是高斯核的一个常见要求。
        #高斯核需要一个中心点，如果核大小是奇数，那么这个中心点就是核的中位数，这样可以方便地对图像进行对称的模糊处理。
        self.blur_horizontal=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(kernel_size,1),stride=1,padding=0,groups=3,bias=False)
        #分组数为 3（意味着每个颜色通道独立卷积）
        self.blur_vertical=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(1,kernel_size),stride=1,padding=0,groups=3,bias=False)
        self.k=kernel_size
        self.r=radius

        self.blur=nn.Sequential(
            nn.ReflectionPad2d(padding=radius),
            #该函数对输入的图像以其边界像素为对称轴做四周的轴对称镜像填充，填充的顺序是左→右→上→下。
            self.blur_horizontal,
            self.blur_vertical
        )

        self.pil_to_tensor=transforms.ToTensor()
        self.tensor_to_pil=transforms.ToPILImage()

    def __call__(self, imgs):
        imgs=self.pil_to_tensor(imgs).unsqueeze(0)  #满足卷积要求
        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)
        # 将一个一维的高斯核数组转换为一个二维张量，其中第一个维度表示通道数（这里是3，对应RGB），第二个维度表示高斯核的长度。
        # 这样，每个通道都会有相同的高斯核，可以用于后续的卷积操作。
        x = x / x.sum()  # 确保高斯核归一化
        x = x.view(3, 1, self.k, 1)  # 为水平卷积调整形状
        y = x.view(3, 1, 1, self.k)  # 为垂直卷积调整形状

        # 将高斯核的值复制到卷积层的权重中
        self.blur_horizontal.weight.data.zero_()  # 先清零
        self.blur_vertical.weight.data.zero_()  # 先清零
        self.blur_horizontal.weight.data.copy_(x)
        self.blur_vertical.weight.data.copy_(y)
        # self.blur_horizontal.weight.data.copy_(src=x.view(3,1,self.k,1))    #Channel,Height,Width,Length
        # self.blur_vertical.weight.data.copy_(src=x.view(3,1,1,self.k))

        with torch.no_grad():
            imgs=self.blur(imgs)
            imgs=imgs.squeeze(0)

        imgs=self.tensor_to_pil(imgs)
        return imgs

class MultiViewInjector:
    def __init__(self,*args):
        self.transforms=args[0]
        self.random_flip=transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample=self.random_flip(sample)

        output=[transform(sample) for transform in self.transforms]
        return output

def get_data_transforms(input_shape,s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=eval(input_shape)[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
                                          transforms.ToTensor()])
    return data_transforms













