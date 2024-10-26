'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''

# device settings
device = 'cuda:0'
# data settings
dataset_path = "../data/mvtec"  # parent directory of datasets
dataset = 'mvtec'  # mvtec|btad'
class_name = "screw"  # dataset subdirectory
modelname = "dual_flow"  # export evaluations/logs with this name

# dataset setting
img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
workers = 4

# network hyperparameters
clamp = 2  # clamping parameter
max_grad_norm = 1e0  # clamp gradients to this norm
n_coupling_blocks = (4, 4)  # higher = more flexible = more unstable
channel_hidden = 768  # * 4 # number of neurons in hidden layers of s-t-networks

use_gamma = True

kernel_sizes = 3

# pos_enc
pos_enc = True
pos_enc_dim = 64  # number of dimensions of positional encoding

extractor = "dino"  # feature dataset name (which was used in 'extract_features.py' as 'export_name')
n_feat = {'dino': 768, 'wide_resnet50_2': 1792, 'efficient': 384, 'resnet18': 1024}[
    extractor]  # dependend from feature extractor

img_size = (224, 224)
crop_size = (224, 224)
input_size = crop_size

top_k = 0.03  # top_k

alpha = 0.5

batch_size = 8  # actual batch size is this value multiplied by n_transforms(_test)

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 100  # total epochs = meta_epochs * sub_epochs
sub_epochs = 2  # evaluate after this number of epochs

# optim
lr_init = 1e-4
lr_decay_milestones = [75, 90]
lr_decay_gamma = 0.33
lr_warmup = True
lr_warmup_from = 0.3
lr_warmup_epochs = 6

# output settings
verbose = True
hide_tqdm_bar = True
save_model = False
viz = False
