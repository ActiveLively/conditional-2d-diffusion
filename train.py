from conditioning import TimeEmbedding, LeNet5, get_image_condition
from UNet import UNet
import torch
from diffusion_components import DiffusionSampler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
#import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def sample_mnist(n_samples: int, dataloader, device = 'cpu'):
    try:
        x0, labels = next(sample_mnist.data_iter)
    except:
        sample_mnist.data_iter = iter(dataloader)
        x0, labels = next(sample_mnist.data_iter)
    #x0 = torch.tensor(x0, dtype = torch.float32, device = device)
    x0 = x0.to(device, dtype=torch.float32)
    labels = labels.to(device)
    return x0, labels
    

if __name__ == '__main__':

    h_size = 128
    timesteps = 1000
    batch_size = 64
    steps = 100000
    beta_schedule = 'cosine'

    condition_channels = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    mnist = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    dataloader = DataLoader(mnist, batch_size = batch_size, shuffle = True)

    sampler = DiffusionSampler(timesteps, beta_schedule).to(device)

    model_unet = UNet(in_channels = 1, out_channels=1).to(device)
    embedder = LeNet5(in_channels = 1, cond_channels = condition_channels).to(device)
    # context_embedder = torch.nn.Embedding(num_embeddings=10, embedding_dim=condition_channels).to(device)

    opt = torch.optim.Adam(list(model_unet.parameters()) + list(embedder.parameters()), lr = 1e-3)

    # context, context_shape = get_image_condition('./context_imgs/cond_diff_context_6.jpg',lenet5, condition_channels, device)
    # print("context_shape: ", context_shape)

    loss_hist = []
    for i in range(steps):
        opt.zero_grad()

        x0, labels = sample_mnist(batch_size, dataloader, device)
        x0 = x0 * 2 - 1 # scale to [-1, 1]
        b = x0.size(0)

        context = embedder(x0).unsqueeze(1) # batch_size x condition_channels -> batch_size x 1 x condition_channels

        t = torch.randint(0, timesteps, (b,), dtype = torch.long, device = device)
        # x0 = sample_target(batch_size, device).to(device)
        ## compute loss
        '''
        To-Do:
        See Algorithm 1 (Training): https://arxiv.org/pdf/2006.11239
        '''
        
        xt, noise = sampler.q_sample(x0, t)
        e = model_unet(xt, t, context)
        loss = F.mse_loss(e, noise)

        loss.backward()
        opt.step()
        print(i, loss.item())
        loss_hist.append(loss.item())

    ## save the trained model weights
    torch.save({'unet': model_unet.state_dict(), 
                'embedder': embedder.state_dict()}, 
                f'model_UNet_{beta_schedule}_{timesteps}timesteps_{batch_size}batch_{steps}steps.pt'
                )

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Train Step')
    ax.set_ylabel('Diffusion Loss')
    ax.plot(loss_hist)
    fig.tight_layout()
    fig.savefig(f'loss_UNet_{beta_schedule}_{timesteps}timesteps_{batch_size}batch_{steps}steps.png', dpi = 300)
    #plt.show()