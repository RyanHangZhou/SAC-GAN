import itertools
import os
import time
import torch

from dataset.dataset import CityscapesDataset
from glob import glob
from torch.utils.data import DataLoader

from network.encoder import E_Transformation, E_Layout
from network import ResNet
from network.discriminator import D_Transformation, D_Layout
from network.generator import SACGAN


class SAC_GAN(object):

    def __init__(self, args):
        self.dataset = args.dataset
        self.dataset_name = args.dataset_name

        self.result_dir = args.result_dir
        self.batch_size = args.batch_size

        self.epoch = args.epoch
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.save_freq = args.save_freq

        """ Weight """
        self.theta_rec_weight = args.theta_rec_weight
        self.affine_weight = args.affine_weight
        self.layout_weight = args.layout_weight

        self.img_h = args.img_h
        self.img_w = args.img_w
        self.patch_s = args.patch_s
        self.class_num = args.class_num
        self.target_class = args.target_class
        self.is_layout_real = args.is_layout_real

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# dataset name: ", self.dataset_name)
        print("# dataset : ", self.dataset)
        print("# result_dir : ", self.result_dir)
        print("# batch_size : ", self.batch_size)
        print("# epoch: ", self.epoch)
        print("# learning rate: ", self.lr)
        print("# weight decay: ", self.weight_decay)
        print("# save frequency: ", self.save_freq)
        print("# image height: ", self.img_h)
        print("# image width: ", self.img_w)
        print("# object patch height/width: ", self.patch_s)
        print("# class number: ", self.class_num)
        print("# target class: ", self.target_class)
        print("# is_layout_real: ", self.is_layout_real)
        print("# theta_rec_weight: ", self.theta_rec_weight)
        print("# affine_weight:", self.affine_weight)
        print("# layout_weight:", self.layout_weight)
    
    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self, args):
        """ DataLoader """
        if self.dataset_name == "Cityscapes":
            self.train_data = CityscapesDataset(self.dataset, self.img_h, self.img_w, self.class_num, self.patch_s, self.is_layout_real, self.target_class, is_train=True)
            self.test_data = CityscapesDataset(self.dataset, self.img_h, self.img_w, self.class_num, self.patch_s, self.is_layout_real, self.target_class, is_train=False)
        elif self.dataset_name == "chair":
            self.train_data = ChairDataset(self.dataset, self.img_h, self.img_w, self.class_num, self.is_layout_real, is_train=True)
            self.test_data = ChairDataset(self.dataset, self.img_h, self.img_w, self.class_num, self.is_layout_real, is_train=False)

        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        """ Define networks """
        self.e_trans = E_Transformation().to(self.device)
        self.e_patch = ResNet.build(in_dim=2, out_dim=args.object_dim).to(self.device)
        self.e_layout = E_Layout().to(self.device)
        self.d_trans = D_Transformation().to(self.device)
        self.d_layout = D_Layout(class_num=self.class_num).to(self.device)
        self.sacgan = SACGAN(self.e_trans, self.e_patch, self.e_layout, self.d_trans, self.d_layout, args).to(self.device)
        
        """ Trainer """
        self.E_optim = torch.optim.Adam(itertools.chain(self.sacgan.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.G_optim = torch.optim.Adam(itertools.chain(self.sacgan.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.d_trans.parameters(), self.d_layout.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)


    def single_build_model(self, args):
        """ Define networks """
        self.e_trans = E_Transformation().to(self.device)
        self.e_patch = ResNet.build(in_dim=2, out_dim=args.object_dim).to(self.device)
        self.e_layout = E_Layout().to(self.device)
        self.d_trans = D_Transformation().to(self.device)
        self.d_layout = D_Layout(class_num=self.class_num).to(self.device)
        self.sacgan = SACGAN(self.e_trans, self.e_patch, self.e_layout, self.d_trans, self.d_layout, args).to(self.device)
        

    def train(self):
        self.sacgan.train(), self.d_trans.train(), self.d_layout.train()

        start_iter = 0
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, 'model'), start_iter)
                print(" [*] Load SUCCESS")

        """ Training loop """
        print('Start training!')

        start_time = time.time()

        step = 0
        iteration = self.epoch * len(self.train_dataloader)
        for step_epoch in range(self.epoch): 
            for target in self.train_dataloader:

                """ 0: reconstruction, 1: generation, 2: discrimination """
                mode_list = [1, 2, 0]

                for mode in mode_list: 
                    if mode==0:
                        self.E_optim.zero_grad()
                    elif mode==1:
                        self.G_optim.zero_grad()
                    elif mode==2:
                        self.D_optim.zero_grad()

                    loss_list = self.sacgan(target)

                    if mode==0: 
                        loss = self.theta_rec_weight*loss_list['rec_loss']
                    elif mode==1:
                        loss = self.theta_rec_weight*loss_list['rec_loss'] + self.affine_weight*loss_list['g_t_loss'] + self.layout_weight*loss_list['g_layout_loss']
                    elif mode==2:
                        loss = self.affine_weight*loss_list['d_t_loss'] + self.layout_weight*loss_list['d_layout_loss']

                    loss.backward()
                    if(mode==0): 
                        self.E_optim.step()
                    elif(mode==1):
                        self.G_optim.step()
                    elif(mode==2):
                        self.D_optim.step()

                    print("[%5d/%5d] time: %4.4f theta_rec_loss: %.8f, d_t_loss: %.8f, g_t_loss: %.8f, d_layout_loss: %.8f, g_layout_loss: %.8f" \
                        % (step, iteration, time.time() - start_time, loss_list['rec_loss'], loss_list['d_t_loss'], loss_list['g_t_loss'], loss_list['d_layout_loss'], loss_list['g_layout_loss']))

                    if step % self.save_freq == 0:
                        self.save(os.path.join(self.result_dir, 'ckpt'), step)

                    step += 1

    def save(self, dir, step):
        params = {}
        params['sacgan'] = self.sacgan.state_dict()
        torch.save(params, os.path.join(dir, 'params_%07d.pt' % step))

    def load(self, dir, step):

        # if torch.cuda.is_available():
        #     map_location=lambda storage, loc: storage.cuda()
        # else:
        #     map_location='cpu'
            
        # checkpoint = torch.load(load_path, map_location=map_location)

        print('Load model: ', os.path.join(dir, 'params_%07d.pt' % step))
        # params = torch.load(os.path.join(dir, 'params_%07d.pt' % step), map_location=torch.device('cpu'))
        params = torch.load(os.path.join(dir, 'params_%07d.pt' % step))
        self.sacgan.load_state_dict(params['sacgan'])


    def test(self):
        model_list = glob(os.path.join(self.result_dir, 'ckpt', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir, 'ckpt'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        with torch.no_grad():
            self.sacgan.eval()
            for target in self.test_dataloader:
                self.sacgan.inference(target, self.result_dir)
    

    def single_test(self, ckpt_path, target, result_dir):
        model_list = glob(os.path.join(ckpt_path, 'ckpt', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(ckpt_path, 'ckpt'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        with torch.no_grad():
            self.sacgan.eval()
            self.sacgan.inference(target, result_dir)

