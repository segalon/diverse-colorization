import os
import random
import argparse

from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt
import wandb
import matplotlib
import functools

torch.manual_seed(12)
matplotlib.use('Agg')


def read_sk(path):
    desired_size = (256, 256)

    img = Image.open(path)
    w, h = img.size

    img_left_area = (0, 0, w / 2, h)
    img_right_area = (w / 2, 0, w, h)

    img_left = img.crop(img_left_area)
    img_right = img.crop(img_right_area)

    img_pd_left = ImageOps.pad(img_left, desired_size)
    img_pd_right = ImageOps.pad(img_right, desired_size)

    return img_pd_right.convert('L'), img_pd_left


class AnimeDataset(Dataset):
    """Anime dataset."""

    def __init__(self, root_dir, gs_transform=None, cl_transform=None, colored_name="colored"):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.gs_transform = gs_transform
        self.cl_transform = cl_transform
        self.name_list = os.listdir(root_dir)
        self.data_size = len(self.name_list)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        colored_img_path = os.path.join(self.root_dir, self.name_list[idx])
        prefix = self.name_list[idx].split("_")[0]
        if prefix == "sk1":
            gray_image, colored_image = read_sk(colored_img_path)
        else:
            colored_image = Image.open(colored_img_path)
        if self.gs_transform:
            gray_image = self.gs_transform(gray_image)
        if self.cl_transform:
            colored_image = self.cl_transform(colored_image)

        return gray_image, colored_image, colored_img_path


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, n_preds=1, kernel_size=4, stride=2, output_padding=(0, 0), input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in x images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=kernel_size,
                             stride=stride, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc * n_preds,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1, output_padding=output_padding)
            down = [downconv]

            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1, bias=use_bias, output_padding=output_padding)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=1, bias=use_bias, output_padding=output_padding)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            out = self.model(x)
            return torch.cat([x, out], 1)


# modified version of https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    # in our case input_nc=1, output_nc=3, num_downs=7?

    def __init__(self, input_nc, output_nc, num_downs, n_preds=1, kernel_size=4, stride=2, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in x images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure

        output_padding = 0
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, kernel_size=kernel_size,
                                             output_padding=output_padding,
                                             stride=stride, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer

        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, kernel_size=kernel_size,
                                                 output_padding=output_padding,
                                                 stride=stride, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
            i += 1
        # gradually reduce the number of filters from ngf * 8 to ngf

        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, kernel_size=kernel_size,
                                             output_padding=output_padding,
                                             stride=stride, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, kernel_size=kernel_size,
                                             output_padding=output_padding,
                                             stride=stride, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, kernel_size=kernel_size,
                                             output_padding=output_padding,
                                             stride=stride, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, n_preds=n_preds, kernel_size=kernel_size,
                                             output_padding=output_padding,
                                             stride=stride, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=4, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in x images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)


def denorm(img):
    denorm_img = ((img + 1) / 2)
    return denorm_img


def pred_and_save(model, dataloader, path, device, img_dim=256):
    print("saving generated images")
    model.eval()
    with torch.no_grad():
        it = iter(dataloader)
        grayscale, colored_real, _ = next(it)
        grayscale = grayscale.to(device)
        img_pred = denorm(model(grayscale))

        img = img_pred
        img = img.reshape(-1, 3, img_dim, img_dim)

        save_image(img, path)

        name = path.split("/")[-1]
        wandb.log({name: wandb.Image(path)})


def get_target_tensor(prediction, target_is_real):
    """Create label tensors with the same size as the x.
  Parameters:
      prediction (tensor) - - tpyically the prediction from a discriminator
      target_is_real (bool) - - if the ground truth label is for real images or fake images
  Returns:
      A label tensor filled with ground truth label, and with the size of the x
  """

    if target_is_real:
        target_tensor = torch.Tensor([1.0])
    else:
        target_tensor = torch.Tensor([0.0])
    return target_tensor.expand_as(prediction)


def generator_loss(colored_real, colored_fake, device, n_preds, H=256, W=256):
    l1_loss = torch.nn.L1Loss()

    # colored_real B x C x H x W
    # colored_fake B x M x C x H x W

    if n_preds > 1:
        # B x C * M x H x W => B x M x C x H x W
        colored_fake = colored_fake.reshape(-1, n_preds, 3, H, W)
        colored_real = colored_real.unsqueeze(1)  # B x C x H x W =>  B x 1 x C x H x W
        diff = torch.abs(colored_real - colored_fake)
        diff_avg = diff.mean(dim=(2, 3, 4))  # B x M
        diff_mins = torch.min(diff_avg, dim=1)[0]  # B
        loss_l1_res = diff_mins.mean()

    else:
        loss_l1_res = l1_loss(colored_fake, colored_real)

    loss_G = loss_l1_res
    return loss_G


def reset_grad(g_opt, d_opt):
    g_opt.zero_grad()


def copy_kernels(G, prev_weights, n_preds=96):
    layer_name = 'model.model.3.weight'

    idxs = list(range(0, n_preds * 3, 3))
    fixed_idxs = []
    changed_idxs = []
    n_copied = 0
    with torch.no_grad():
        prev_weights = prev_weights.cuda()
        state_dict = G.state_dict()
        for idx in idxs:
            cur_kernels = state_dict[layer_name][:, idx: idx + 3, :, :]
            if cur_kernels.equal(prev_weights[:, idx: idx + 3, :, :]):
                # these kernels haven't changed, and so they haven't been used at all in the epoch
                fixed_idxs.append(idx)
            else:
                changed_idxs.append(idx)

        for f_idx in fixed_idxs:
            if len(changed_idxs) > 0:
                idx_cp_from = random.sample(changed_idxs, 1)[0]
                state_dict[layer_name][:, f_idx: f_idx + 3, :, :] = state_dict[layer_name][:,
                                                                    idx_cp_from: idx_cp_from + 3, :, :]
                n_copied += 1
            else:
                print("len(changed_idxs)==0")

    G.load_state_dict(state_dict)
    print("Copied {} kernels".format(n_copied))


def create_transforms():
    transform_list = [transforms.ToTensor()]
    gray_transforms = transforms.Compose(transform_list + [transforms.Normalize((0.5,), (0.5,))])
    color_transforms = transforms.Compose(transform_list + [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return gray_transforms, color_transforms


def train_loop(training_data, val_data, epochs_num, G, g_optimizer, gen_loss, batch_size,
               preds_save_dir, pth_path, n_preds=1, start_epoch=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    with torch.no_grad():
        layer_name = 'model.model.3.weight'
        prev_weights = G.state_dict()[layer_name].clone()

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    # Start training
    total_step = len(train_dataloader)

    g_list_losses = []

    G = G.to(device)
    cur_instance_epoch = 0
    for epoch in range(start_epoch, epochs_num):
        G.train()

        running_g_loss = 0.0
        for i, (grayscale, colored_real, _) in enumerate(train_dataloader):
            grayscale = grayscale.to(device)
            colored_real = colored_real.to(device)
            colored_fake = G(grayscale)

            g_loss = gen_loss(colored_real, colored_fake, device, n_preds=n_preds)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            running_g_loss += g_loss

            print_every = 2
            if (i + 1) % print_every == 0:
                print("epoch:", epoch)
                print("step: [{}/{}]".format(i, total_step))
                print("G loss:", g_loss)
                print("-" * 10)

        if (epoch + 1) % 50 == 0:
            torch.save(G.state_dict(), os.path.join(pth_path, "G_mpreds_{}.pth".format(epoch)))

        torch.save(G.state_dict(), os.path.join(pth_path, "G_mpreds.pth"))
        wandb.log({"G train loss": running_g_loss})

        g_list_losses.append(running_g_loss)

        disp_batch_size = 1
        idxs_val = random.sample(range(0, len(val_data)), 1)
        print("idxs_val:")
        print(idxs_val)
        val_subset_data = torch.utils.data.Subset(val_data, idxs_val)
        val_dataloader = DataLoader(val_subset_data, disp_batch_size, shuffle=False)

        path = os.path.join(preds_save_dir, "images_val_epoch={}_i={}.png".format(epoch, i))
        pred_and_save(G, val_dataloader, path, device)

        idxs_train = random.sample(range(0, len(training_data)), 1)

        train_subset_data = torch.utils.data.Subset(training_data, idxs_train)
        train_sub_dataloader = DataLoader(train_subset_data, disp_batch_size, shuffle=False)
        path = os.path.join(preds_save_dir, "images_train_epoch={}_i={}.png".format(epoch, i))
        pred_and_save(G, train_sub_dataloader, path, device)

        copy_kernels(G, prev_weights, n_preds=n_preds)

        with torch.no_grad():
            layer_name = 'model.model.3.weight'
            prev_weights = G.state_dict()[layer_name].clone()

        cur_instance_epoch += 1


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--train_path', type=str, default="data/train_data")
    parser.add_argument('--val_path', type=str, default="data/val_data")
    parser.add_argument('--preds_path', type=str, default="/generated_images/")
    parser.add_argument('--G_path', type=str, default="")
    parser.add_argument('--D_path', type=str, default="")
    parser.add_argument('--start_epoch', type=int, default=0)

    return parser


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


if __name__ == "__main__":
    wandb.init(project='colorization_n_preds_evol', entity='bkw1')
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    args.train_path = ""
    args.val_path = ""
    args.preds_path = ""

    gs_trans, color_trans = create_transforms()

    training_data = AnimeDataset(args.train_path, gs_transform=gs_trans, cl_transform=color_trans)
    val_data = AnimeDataset(args.val_path, gs_transform=gs_trans, cl_transform=color_trans)

    epochs_num = 500

    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999

    batch_size = 16
    n_preds = 32
    n_layers = 8
    ngf = 130

    preds_save_path = args.preds_path
    start_epoch = args.start_epoch

    gen_loss = generator_loss

    print("ngf:", ngf)
    print("batch_size:", batch_size)

    G = UnetGenerator(1, 3, n_layers, kernel_size=4, stride=2, ngf=ngf, n_preds=n_preds)
    if args.G_path == "":
        print("New G")
    else:
        print("Loading G")
        G.load_state_dict(torch.load(args.G_path))

    pytorch_total_params = sum(p.numel() for p in G.parameters())
    print("number of model parameters:", pytorch_total_params)

    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

    train_loop(training_data, val_data, epochs_num, G, g_optimizer, gen_loss, batch_size, args.preds_path,
               n_preds=n_preds,
               start_epoch=args.start_epoch, pth_path="")
