import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from conditioning import get_image_condition, LeNet5


from train import DiffusionSampler
from UNet import UNet

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #x0 = torchvision.io.read_image('dog.jpeg').permute(1, 2, 0).rot90(k = 3).float().unsqueeze(0) / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # [0,1] -> [-1,1]
    ])
    mnist_data = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )


    idx = torch.randint(0, len(mnist_data), (1,)).item()
    img, _ = mnist_data[idx]
    x0 = img.permute(1, 2, 0).unsqueeze(0).float()

    timesteps = 1000
    cosine_sampler = DiffusionSampler(timesteps, 'cosine').to(device)
    # linear_sampler = DiffusionSampler(timesteps, 'scaled_linear').to(device)

    ##########set UNet + LeNet5 model
    unet_cosine = UNet(in_channels=1, out_channels=1).to(device)
    lenet_embedder = LeNet5(in_channels=1, cond_channels=128).to(device)
    checkpoints = torch.load('models/model_UNet_cosine_1000timesteps_64batch_10000steps.pt', map_location=device)
    unet_cosine.load_state_dict(checkpoints['unet'])
    lenet_embedder.load_state_dict(checkpoints['embedder'])

    context, context_shape = get_image_condition(
        './context_imgs/cond_diff_context_6.jpg', 
        lenet_embedder, 
        condition_channels = 128, 
        device = device
    )
    context = context.unsqueeze(1)
    
    # unet_linear.load_state_dict(torch.load('results/model_UNet_scaled_linear_1000timesteps_256batch_20000steps.pt', map_location=device))
    unet_cosine.eval()
    lenet_embedder.eval()

    num_steps = 10
    steps = torch.linspace(0, timesteps-1, num_steps, dtype = torch.long)
    xT = torch.randn(1, 1, 32, 32, device=device)
    with torch.no_grad():
            xt_cosine, _ = cosine_sampler.p_sample_loop(unet_cosine, xT, context, num_steps)
            # xt_linear, _ = linear_sampler.p_sample_loop(unet_linear, xT, context, num_steps)

    fig, ax = plt.subplots(2, num_steps, figsize = (num_steps, 3))

    for i in range(num_steps):
        steps

        ax[0, i].axis('off')
        ax[1, i].axis('off')
        ax[0, i].set_title(f'Step: {steps[i].item()}', fontsize = 10)

        xt_cosine_out = xt_cosine[i]
        # xt_linear_out = xt_linear[i]

        # reshape to H x W instead of 1 x H x W
        img_cos = (xt_cosine_out[0, 0].detach().cpu() + 1) / 2
        # img_lin = (xt_linear_out[0, 0].detach().cpu() + 1) / 2

        ax[0, i].imshow(img_cos)
        # ax[1, i].imshow(img_lin)
        print(i)

    fig.tight_layout()
    fig.savefig(f'mnist_reverse_process_{timesteps}steps_long_context_6.png', dpi = 300)


    