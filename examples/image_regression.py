import argparse
import torch
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from neureps import modules, encodings, utils


def learn_image(image_path: str,
                activation: str,
                lr: float,
                steps: int,
                log_every: int = 100,
                device='cuda'):
    image_path = Path(image_path)
    image = Image.open(image_path)
    num_channels = len(image.getbands())

    target = to_tensor(image).to(device)

    # Add a channel axis for gray-scale images
    if num_channels == 1:
        target = target.unsqueeze(-1)

    model = modules.ImplicitNeuralRepresentation(
        encoder=encodings.IdentityPositionalEncoding(in_dim=2),
        mlp=modules.MLP(
            in_dim=2,
            out_dim=num_channels,
            hidden_dims=[256] * 5,
            block_factory=utils.get_block_factory(activation),
            final_activation=torch.sigmoid
        )
    ).to(device)

    grid = utils.meshgrid(image.height, image.width).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        opt.zero_grad()

        out = model(grid)
        loss = F.mse_loss(out, target)
        loss.backward()

        opt.step()

        if log_every and not (step % log_every):
            print(f'{step:05d}/{steps:05d} mse: {loss:.04f}, psnr: {utils.psnr(loss):.4f}')
            save_image(out.squeeze(), image_path.with_name(image_path.stem + '_recon' + image_path.suffix))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('--activation', default='siren')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    learn_image(**vars(args))
