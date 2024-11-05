# coding=utf8
# 你配不上自己的野心，也就辜负了先前的苦难
# 整段注释Control+/
import torch.nn
import torchvision
from torch import nn

#debug1:pretrained 调整为weights参数


class MLPhead(nn.Module):
    def __init__(self,in_channels,mlp_hidden_size,projection_size):
        super(MLPhead, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(in_features=in_channels,out_features=mlp_hidden_size),
            nn.BatchNorm1d(num_features=mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_hidden_size,out_features=projection_size)
        )

    def forward(self,x):
        return self.net(x)

class ResNet18(nn.Module):
    def __init__(self,*args,**kwargs):
        super(ResNet18, self).__init__()
        if kwargs['name']=='resnet18':
            resnet=torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        elif kwargs['name']=='resnet50':
            resnet=torchvision.models.resnet50(weights=torchvision.models.ResNet18_Weights.DEFAULT)

        self.encoder=torch.nn.Sequential(*list(resnet.children())[:-1])
        #resnet.children()：这个方法会返回一个迭代器，包含 ResNet 模型中所有的子模块（即所有的层）
        #[:-1]：这是一个切片操作，它取出列表中的所有元素，除了最后一个元素。在 ResNet 模型中，最后一个元素通常是全连接层（fc层），所以这里我们不包括它
        self.projection=MLPhead(in_channels=resnet.fc.in_features,**kwargs['projection_head'])
        #resnet.fc.in_features：这是 ResNet 模型中最后一个全连接层的输入特征数，即前一层的输出特征数
        #kwargs['projection_head'] 应该是一个字典，其中包含了 mlp_hidden_size 和 projection_size 的值

    def forward(self,x):
        head=self.encoder(x)
        head=head.view(head.shape[0],head.shape[1])
        #这个操作通常在将卷积层的输出传递给全连接层之前进行，因为全连接层需要二维输入。通过这种方式，卷积层的多维输出被展平为一维，以便全连接层可以处理。
        head=self.projection(head)
        return head






