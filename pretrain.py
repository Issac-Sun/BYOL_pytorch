# coding=utf8
# 你配不上自己的野心，也就辜负了先前的苦难
# 整段注释Control+/
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
from shutil import copyfile

from torchvision import datasets

from BYOL.data import get_data_transforms, MultiViewInjector
from BYOL.model import ResNet18, MLPhead

#debug1:添加了weight_only
#debug2:保护主函数
def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))


class BYOLTrainer:
    def __init__(self,online_network,target_network,predictor,optimizer,device,**params):
        self.online_network=online_network
        self.target_network=target_network
        self.optimizer=optimizer
        self.predictor=predictor
        self.max_eps=params['max_epochs']
        self.writer=SummaryWriter(log_dir='logs')
        self.momentum=params['m']
        self.device=device
        self.batch_size=params['batch_size']
        self.num_workers=params['num_workers']
        self.checkpoint=params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./configs.yaml", "pretrain.py"])

    @torch.no_grad()
    def _update_target_parameters(self):
        """
             Momentum update of the key encoder
             """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @staticmethod
    def regression_loss(x,y):
        # 计算两个向量的点积
        dot_product = torch.sum(x*y,dim=-1)
        # 计算两个向量的欧几里得范数
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)+ 1e-8
        norm_y = torch.norm(y, p=2, dim=-1, keepdim=True)+ 1e-8
        # 计算余弦相似度
        cosine_similarity = dot_product / (norm_x * norm_y)
        # 根据公式计算损失
        loss = 2 - 2 * cosine_similarity
        return loss

        #------Former@repo:sthalles/PyTorch-BYOL -------
        # def regression_loss2(x, y):
        #     x = F.normalize(x, dim=1)
        #     y = F.normalize(y, dim=1)
        #     return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        step = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()

        for epoch_counter in range(self.max_eps):

            for (batch_view_1, batch_view_2), _ in train_loader:

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)

                if step/20 == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=step)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=step)

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=step)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print("loss={}".format(loss))
                self._update_target_parameters()  # update the key encoder
                step += 1

            print("End of epoch {}".format(epoch_counter))
        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        """原文中提出了symmetrize的概念"""

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)

torch.manual_seed(seed=28)

def main():
    config = yaml.load(open("./configs.yaml", "r",encoding='utf-8'), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_data_transforms(**config['data_transforms'])

    # train_dataset = datasets.STL10('./root', split='train+unlabeled', download=True,
    #                                transform=MultiViewInjector([data_transform, data_transform]))

    #---CIFAR10---
    train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                     transform=MultiViewInjector([data_transform, data_transform]))

    # online network
    online_network = ResNet18(**config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
            map_location=torch.device(device),weights_only=True)

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPhead(in_channels=online_network.projection.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ResNet18(**config['network']).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    trainer.train(train_dataset)

if __name__ == '__main__':
    main()
