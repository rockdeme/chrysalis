import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import pytorch_lightning as L
import matplotlib.pyplot as plt
from itertools import islice
from torchinfo import summary
from torchvision import transforms
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint


# setting the seed
L.seed_everything(42)

# ensure that all operations are deterministic on GPU for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

data_path = "/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/data/xenium_human_breast_cancer/tiles_299.npy"

class NpDataset(data.Dataset):
    def __init__(self, data_file=None, transform=None):
        self.data = np.load(data_file)
        if self.data.shape[-1] == 3:
            self.data = self.data.transpose(0, 3, 1, 2)
        self.data = self.data / 255.0
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()
        if self.transform:
            x = self.transform(x)
        return x, x

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
ds = NpDataset(data_file=data_path, transform=transform)

train_set, val_set = torch.utils.data.random_split(ds, [3510, 396])

train_loader = data.DataLoader(ds, batch_size=1, shuffle=True, drop_last=False, pin_memory=True, num_workers=1)


def show_image(data_loader):
    dataiter = iter(data_loader)
    images = dataiter.next()
    img = images[0][0].clone()  # clone to avoid changing original tensor
    img = img * 0.5 + 0.5
    img_arr = img.numpy().transpose(1, 2, 0)
    plt.imshow(img_arr)
    plt.show()


show_image(train_loader)

class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            # 256x256 => 128x128
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            # 128x128 => 64x64
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            # 64x64 => 32x32
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Flatten(),  # image grid to single feature vector
            nn.Linear(32 * 32 * 2 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * c_hid * 32 * 32), act_fn())
        self.net = nn.Sequential(
            # 32x32=>64x64
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            act_fn(),
            # 64x64 => 128x128
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            act_fn(),
            # 128x128 => 256x256
            nn.ConvTranspose2d(num_input_channels, num_input_channels, kernel_size=3, output_padding=1, padding=1,
                               stride=2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], 2 * 32, 32, 32)
        x = self.net(x)
        return x


class Autoencoder(L.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 32,
        height: int = 32,
    ):
        super().__init__()
        # saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """the forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)


model = Autoencoder(base_channel_size=32, latent_dim=512, width=256, height=256)

summary(model, (1, 3, 256, 256))

latent_dim = 512

trainer = L.Trainer(default_root_dir="/mnt/c/Users/demeter_turos/PycharmProjects/deep_learning/"
                                     "autoencoder_he_image/models/",
                    accelerator="auto",
                    devices=1,
                    max_epochs=100,
                    callbacks=[ModelCheckpoint(save_weights_only=True),
                               # GenerateCallback(get_train_images(8), every_n_epochs=10),
                               LearningRateMonitor("epoch")],
                    limit_val_batches=0,
                    num_sanity_val_steps=0)

trainer.fit(model, train_loader)

model = Autoencoder.load_from_checkpoint("/mnt/c/Users/demeter_turos/PycharmProjects/deep_learning/"
                                         "autoencoder_he_image/models/lightning_logs/version_8/checkpoints/"
                                         "epoch=99-step=390600.ckpt")

data_iter = iter(train_loader)  # Create an iterator for the dataloader


def show_images(model, data_iter):
    model.to(device)
    model.eval()
    num_elements = 9
    elements = [x[0] for x in islice(data_iter, num_elements)]  # Select only the first tensor of each pair

    # Now stack the list of tensors to a single tensor
    elements = torch.cat(elements)

    images = data_iter.next()
    elements = elements.to(device)
    # images =  images[None, :]
    outputs = model(elements)

    outputs = outputs.cpu().detach().numpy()


    fig, axs = plt.subplots(3, 6, figsize=(12, 6))
    axs = axs.flatten()
    for a in axs:
        a.axis('off')

    for idx, i in enumerate([3, 4, 5, 9, 10, 11, 15, 16, 17]):
        out_img = outputs[idx]
        out_img = out_img * 0.5 + 0.5
        out_img_arr = out_img.transpose(1, 2, 0)
        axs[i].imshow(out_img_arr)

    for idx, i in enumerate([0, 1, 2, 6, 7 ,8, 12, 13, 14]):
        img = elements[idx].clone()
        img = img * 0.5 + 0.5
        img_arr = img.cpu().numpy().transpose(1, 2, 0)
        axs[i].imshow(img_arr)


show_images(model, data_iter)
plt.show()

model.eval()

all_encoded_features = []

eval_loader = data.DataLoader(ds, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=1)

with torch.no_grad():
    for batch in eval_loader:
        x, _ = batch
        x = x.to(device)
        encoded_features = model.encoder(x)
        encoded_features = encoded_features.cpu().detach().numpy()
        all_encoded_features.append(encoded_features)

all_encoded_features = np.concatenate(all_encoded_features)

np.save("/mnt/c/Users/demeter_turos/PycharmProjects/chrysalis/data/xenium_human_breast_cancer/aenc_features_512_v2.npy",
        all_encoded_features)
