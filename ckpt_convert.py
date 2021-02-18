import torch
import argparse
import pytorch_lightning as pl

from simclr import SimCLR
from torchvision import models
from simclr.modules import NT_Xent

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--ckpt_path', type=str, required=True,
                    help='pytorch lightning checkpoint path')

parser.add_argument('--save_path', type=str, required=True,
                    help='path to save pytorch checkpoint')

parser.add_argument('--arch', type=str, required=True,
                    help='model arch')

args = parser.parse_args()

arch_dict = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101
}


class ContrastiveLearning(pl.LightningModule):
    def __init__(self, net):
        super().__init__()

        # initialize ResNet
        self.encoder = net
        self.n_features = net.fc.in_features  # get dimensions of fc layer
        self.model = SimCLR(net, 256, self.n_features)
        self.criterion = NT_Xent(
            32, 0.5, world_size=1
        )

    def forward(self, x_i, x_j):
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        return loss

    def training_step(self, batch, batch_idx):
        (x_i, x_j) = batch
        loss = self.forward(x_i, x_j)
        return loss

    def configure_criterion(self):
        criterion = NT_Xent(32, self.hparams.temperature)
        return criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        return optimizer


def convert_model(ckpt_path, save_path, arch):
    net = arch(pretrained=False)

    model = ContrastiveLearning(net)

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])

    torch.save(net.state_dict(), save_path)


convert_model(args.ckpt_path, args.save_path, arch_dict[args.arch])
