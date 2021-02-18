import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7,1,2,3,4,5,0,6"
import argparse
import torch
from PIL import Image
from pathlib import Path
import pytorch_lightning as pl

from torchvision import models
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from simclr import SimCLR
from simclr.modules import NT_Xent

# SimCLR

parser = argparse.ArgumentParser(description='simclr-pytorch-lightning')

parser.add_argument(
    '--image_folder',
    type=str,
    required=True,
    help='path to your folder of images for self-supervised learning')

args = parser.parse_args()

# Encoder
ENCODER = models.resnet18(pretrained=True)

BATCH_SIZE = 64
EPOCHS = 300
LR = 3e-4
NUM_GPUS = 8
IMAGE_SIZE = 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
TEMPERATURE = 0.5
PROJ_DIM = 256
NUM_WORKERS = NUM_GPUS  # multiprocessing.cpu_count()


class ContrastiveLearning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # initialize ResNet
        self.encoder = ENCODER
        self.n_features = ENCODER.fc.in_features  # get dimensions of fc layer
        self.model = SimCLR(ENCODER, PROJ_DIM, self.n_features)
        self.criterion = NT_Xent(
            BATCH_SIZE, TEMPERATURE, world_size=1
        )

    def forward(self, x_i, x_j):
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        return loss

    def training_step(self, batch, batch_idx):
        (x_i, x_j) = batch
        loss = self.forward(x_i, x_j)
        self.log(
            'loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True)
        return loss

    def configure_criterion(self):
        criterion = NT_Xent(BATCH_SIZE, self.hparams.temperature)
        return criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        return optimizer


class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = T.Compose(
            [
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.3),
                T.RandomGrayscale(p=0.2),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2),
                T.RandomResizedCrop((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225])),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')

        return self.transform(img), self.transform(img)


if __name__ == "__main__":
    ds = ImagesDataset(args.image_folder, IMAGE_SIZE)

    train_loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )

    cl = ContrastiveLearning()

    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name='logs'
    )

    checkpoint_callback = ModelCheckpoint(
        period=10,
        save_top_k=-1
    )

    trainer = pl.Trainer(
        gpus=NUM_GPUS,
        distributed_backend='ddp',
        max_epochs=EPOCHS,
        accumulate_grad_batches=1,
        callbacks=[checkpoint_callback],
        logger=logger
        # resume_from_checkpoint = '*.ckpt'
    )

    trainer.sync_batchnorm = True
    trainer.fit(cl, train_loader)
