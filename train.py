from conditioning import TimeEmbedding, LeNet5
from UNet import UNet
import torch
from diffusion_components import DiffusionSampler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image


def sample_misnt(n_samples: int, dataloader, device = 'cpu'):
    try:
        x0, _ = next(sample_misnt.data_iter)
    except:
        sample_misnt.data_iter = iter(dataloader)
        x0, _ = next(sample_misnt.data_iter)
    #x0 = torch.tensor(x0, dtype = torch.float32, device = device)
    x0 = x0.to(device, dtype=torch.float32)
    return x0

def get_condition(file_path: str, device = 'cpu'):
    transform = transforms.Compose([
        transforms.Resize((32,32)), 
        transforms.ToTensor()
    ])
    img = Image.open(file_path).convert('L')
    img_resized = transform(img).unsqueeze(0).to(device)
    lenet5 = LeNet5(in_channels = 1, cond_channels = h_size).to(device)
    lenet5.to(device)
    lenet5.eval()

    with torch.no_grad():
        context = lenet5(img_resized)

    return context, context.shape 

    

if __name__ == '__main__':

    h_size = 128
    timesteps = 1000
    batch_size = 256
    steps = 20000
    beta_schedule = 'cosine'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])

    mnist = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    dataloader = DataLoader(mnist, batch_size = batch_size, shuffle = True)

    sampler = DiffusionSampler(timesteps, beta_schedule).to(device)

    model_unet = UNet(in_channels = 1).to(device)
    opt = torch.optim.Adam(model_unet.parameters(), lr = 1e-3)

    context, context_shape = get_condition('./data/sample_condition.png', device)

    print("context_shape: ", context_shape)

    loss_hist = []
    for i in range(steps):
        opt.zero_grad()

        x0 = sample_misnt(batch_size, dataloader, device)
        x0 = x0 * 2 - 1 # scale to [-1, 1]
        b = x0.size(0)

        t = torch.randint(0, timesteps, (b,), dtype = torch.long, device = device)
        # x0 = sample_target(batch_size, device).to(device)
        ## compute loss
        '''
        To-Do:
        See Algorithm 1 (Training): https://arxiv.org/pdf/2006.11239
        '''
        xt, noise = sampler.q_sample(x0, t)
        e = model_unet(xt, context, t)
        loss = F.mse_loss(e, noise)

        loss.backward()
        opt.step()
        print(i, loss.item())
        loss_hist.append(loss.item())

    ## save the trained model weights
    torch.save(model_unet.state_dict(), f'model_UNet_{beta_schedule}_{timesteps}timesteps_{batch_size}batch_{steps}steps.pt')

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Train Step')
    ax.set_ylabel('Diffusion Loss')
    ax.plot(loss_hist)
    fig.tight_layout()
    fig.savefig(f'loss_UNet_{beta_schedule}_{timesteps}timesteps_{batch_size}batch_{steps}steps.png', dpi = 300)
    #plt.show()