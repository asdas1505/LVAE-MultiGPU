import torch
from torchvision.utils import save_image
import pickle
from experiment import LVAEExperiment
import matplotlib.pyplot as plt

def gaussian_noise(input, std=0.0):
    noise = torch.normal(mean=0, std=torch.ones_like(input)*std)
    return input + noise

def grad_switch(params, grad_bool=False):
    for param in params:
        param.requires_grad = grad_bool

args_pkl_path = '/home/lsinghal/new/Iterative_Models/ladder-vae-pytorch/output/210918_092325_cifar10,3ly,4bpl,64ch,skip,gate,block=bacdbacd,elu,freeb=0.5,drop=0.2,learnp,seed42/checkpoints/config.pkl'
checkpoint_path = '/home/lsinghal/new/Iterative_Models/ladder-vae-pytorch/output/210918_092325_cifar10,3ly,4bpl,64ch,skip,gate,block=bacdbacd,elu,freeb=0.5,drop=0.2,learnp,seed42/checkpoints/model_2500000.pt'

experiment = LVAEExperiment()

with open(args_pkl_path, 'rb') as file:
    args = pickle.load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
experiment._device = device

dataloaders = experiment._make_datamanager(args_eval=args)

model = experiment._make_model(args_eval=args, dataloader=dataloaders)

model.load_state_dict(torch.load(checkpoint_path))

grad_switch(model.parameters(), grad_bool=False)


print('Model: Ready!!')
#######################################
# Dataloader
train_loader = dataloaders.train
print("Dataloader: Ready!!!")

print('eval: start')


# /cifs/data/tserre/CLPS_Serre_Lab/projects



save_img_path = '/home/lsinghal/new/Iterative_Models/ladder-vae-pytorch/output_iter/CIFAR_noisy/'
                # 'out_refined.png'
for idx, (X_train, y_train) in enumerate(train_loader):
    # output = model.forward(X_train.to(device))
    break


for std in [0.1]:
    X_noise_input = gaussian_noise(X_train, std=std)
    X_noise_input = torch.clip(X_noise_input, 0, 1)
    X_rec, elbo_refined, loss_list, elbo_list, mse_list = model.forward_evaluate(X_noise_input.to(device), eval_iters=5000)
    save_image(X_noise_input, save_img_path + 'noisy_img5_{}.png'.format(std))
    save_image(X_rec, save_img_path + 'rec_noisy_img5_{}.png'.format(std))
    plt.plot(loss_list)
    plt.xlabel('Number of steps')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()

print('eval: end')


# save_noise_img_path = '/home/lsinghal/new/Iterative_Models/ladder-vae-pytorch/output_iter/CIFAR_noisy/noise_img.png'
# save_noise_rec_img_path = '/home/lsinghal/new/Iterative_Models/ladder-vae-pytorch/output_iter/CIFAR_noisy/noise_rec_img.png'



# plt.savefig('/home/lsinghal/new/Iterative_Models/ladder-vae-pytorch/output_iter/CIFAR/ELBO_trend.png')
# plt.savefig('/home/lsinghal/new/Iterative_Models/ladder-vae-pytorch/output_iter/CIFAR/loss_check.png')









# experiment.load_args_from_pickle(args_pkl_path)
# experiment.load_model(checkpoint_path, step=900000)