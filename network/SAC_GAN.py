from ast import arg
import itertools, time
from glob import glob
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage.transform import resize

from dataset import dataset_cityscapes
from dataset import dataset_chair
from utils import util
from network.generator import *
from network.discriminator import *
from network.Canny import CannyEdgeDetector

class SAC_GAN(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.dataset_name = args.dataset_name

        self.result_dir = args.result_dir
        self.batch_size = args.batch_size

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.decay_flag = args.decay_flag

        """ Weight """
        self.rec_weight = args.rec_weight
        self.latent_rec_weight = args.latent_rec_weight
        self.affine_weight = args.affine_weight
        self.layout_weight = args.layout_weight

        self.img_h = args.img_h
        self.img_w = args.img_w
        self.patch_h = args.patch_h
        self.patch_w = args.patch_w
        self.class_num = args.class_num
        self.target_class = args.target_class
        self.layout_flag = args.layout_flag

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
        print("# iteration: ", self.iteration)
        print("# learning rate: ", self.lr)
        print("# weight decay: ", self.weight_decay)
        print("# print frequency: ", self.print_freq)
        print("# save frequency: ", self.save_freq)
        print("# image height: ", self.img_h)
        print("# image width: ", self.img_w)
        print("# object patch height: ", self.patch_h)
        print("# object patch width: ", self.patch_w)
        print("# class number: ", self.class_num)
        print("# target class: ", self.target_class)
        print("# layout_flag: ", self.layout_flag)
        print("# rec_weight: ", self.rec_weight)
        print("# latent_rec_weight: ", self.latent_rec_weight)
        print("# affine_weight:", self.affine_weight)
        print("# layout_weight:", self.layout_weight)
    
    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """

        if self.dataset_name=="cityscapes":
            self.train_data = dataset_cityscapes.ImageDataset(self.dataset, self.img_h, self.img_w, self.class_num, self.layout_flag, is_train=True)
            self.test_data = dataset_cityscapes.ImageDataset(self.dataset, self.img_h, self.img_w, self.class_num, self.layout_flag, is_train=False)
        elif self.dataset_name=="chair":
            self.train_data = dataset_chair.ImageDataset(self.dataset, self.img_h, self.img_w, self.class_num, self.layout_flag, is_train=True)
            self.test_data = dataset_chair.ImageDataset(self.dataset, self.img_h, self.img_w, self.class_num, self.layout_flag, is_train=False)

        """ Define Generator, Discriminator """
        
        self.stnet = STNet().to(self.device)
        self.enc_t = TEncoderNet().to(self.device)
        self.dis_t = TDisNet().to(self.device)
        self.dis_sem = SemDisNet().to(self.device)

        """ Define Loss """
        self.BCE_loss = nn.BCELoss().to(self.device)

        """ Trainer """
        self.E_optim = torch.optim.Adam(itertools.chain(self.enc_t.parameters(), self.stnet.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.G_optim = torch.optim.Adam(itertools.chain(self.stnet.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.dis_t.parameters(), self.dis_sem.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

 
    def train(self):
        self.stnet.train(), self.enc_t.train(), self.dis_t.train(), self.dis_sem.train()

        writer = SummaryWriter(log_dir=os.path.join(self.result_dir, 'runs'))
        start_iter = 0
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, 'model', '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, 'model'), start_iter)
                print(" [*] Load SUCCESS")

        # training loop
        print('training start !')
        start_time = time.time()
        step = 0
        for step_epoch in range(self.epoch): 
            for object_image, object_mask, back_image, layout, object_mask_ref, layout_ref, indx in self.train_data:

                # extract object patch, mask and theta_gt
                theta_gt, patch_mask, patch_obj, norm_mask, norm_obj = self.object_extract(object_mask, object_image, indx)
                theta_gt_ref, _, _, _, _ = self.object_extract(object_mask_ref, object_image, indx)
                # edge extraction
                object_edge = self.edge_extract(norm_obj, indx)

                ## To cuda ##
                # raw patch size
                _, _, local_h, local_w = np.shape(patch_mask)
                object_mask = torch.unsqueeze(object_mask, dim=0).to(self.device)
                layout = torch.unsqueeze(layout, dim=0).to(self.device)
                back_image = torch.unsqueeze(back_image, dim=0).to(self.device)
                layout_ref = torch.unsqueeze(layout_ref, dim=0).to(self.device)
                patch_obj = patch_obj.to(self.device)
                patch_mask = patch_mask.to(self.device)

                iterlist = [1, 2, 0]
                for flag in iterlist: 
                    if(flag==0):
                        self.E_optim.zero_grad()
                    elif(flag==1):
                        self.G_optim.zero_grad()
                    elif(flag==2):
                        self.D_optim.zero_grad()

                    ### NETWORK
                    # VAE -> encode theta_gt and reconstruct
                    theta_mu, theta_logvar = self.enc_t(theta_gt)
                    theta_reparam = self.re_param(theta_mu, theta_logvar, mode='train')
                    theta_reparam_expand = theta_reparam.expand([-1, -1, self.img_h, self.img_w])
                    layout_n_z = torch.cat([layout, theta_reparam_expand], 1)
                    kl_loss = torch.mean(-0.5 * torch.sum(1 + theta_logvar - theta_mu ** 2 - theta_logvar.exp(), dim = 1), dim = 0)

                    theta_rec, _, object_mask_rec, z_dec, layout_dec, object_dec = self.stnet(layout_n_z, norm_mask, object_edge, self.result_dir, indx) # object + layout -> rec theta


                    # Gen
                    latent_z = torch.FloatTensor(1, 4, 1, 1).normal_(0, 1)
                    latent_z = Variable(latent_z).to(self.device)
                    latent_z_expand = latent_z.expand([-1, -1, self.img_h, self.img_w])
                    layout_n_z_gen = torch.cat([layout, latent_z_expand], 1)
                    theta_gen, _, object_mask_gen, z_gen, layout_gen, object_gen = self.stnet(layout_n_z_gen, norm_mask, object_edge, self.result_dir, indx)

                    layout_ref_n_z_gen = torch.cat([layout_ref, latent_z_expand], 1)
                    theta_gen_ref, _, obj_mask_ref_gen, z_gen2, layout_gen2, object_gen2 = self.stnet(layout_ref_n_z_gen, norm_mask, object_edge, self.result_dir, indx) # object + ref_layout -> gen theta

                    # obtain layout
                    layout_gt = layout * (1. - object_mask) + self.pad_to_nclass(object_mask) # real/anchor
                    layout_ref_gen = layout_ref * (1. - obj_mask_ref_gen) + self.pad_to_nclass(obj_mask_ref_gen) # negative
                    layout_gen = layout * (1. - object_mask_gen) + self.pad_to_nclass(object_mask_gen) # positive

                    if flag ==2: 
                        # save layout
                        util.save_layout(layout_gt, self.result_dir, 'train_original_layout', indx)
                        util.save_layout(layout_ref_gen, self.result_dir, 'train_fake_layout', indx)
                        util.save_layout(layout_gen, self.result_dir, 'train_generated_layout', indx)


                    # discriminators and losses
                    d_theta_gt = self.dis_t(theta_gt)
                    d_theta_rec = self.dis_t(theta_rec)
                    d_theta_gen = self.dis_t(theta_gen)
                    d_theta_gen_ref = self.dis_t(theta_gen_ref)

                    # reconstruction loss
                    weights = [[1*np.ones(self.batch_size), 1*np.ones(self.batch_size), 1*np.ones(self.batch_size)], 
                               [1*np.ones(self.batch_size), 1*np.ones(self.batch_size), 1*np.ones(self.batch_size)]]
                    weights = np.moveaxis(weights, -1, 0) # (4, 2, 3)
                    weights = torch.tensor(weights).to(self.device)
                    # theta_rec_loss = torch.mean(torch.abs(theta_rec - theta_gt))

                    theta_rec_loss = torch.mean(torch.abs(weights*theta_rec - weights*theta_gt))
                    latent_rec_loss = torch.mean(torch.abs(theta_reparam - z_dec)) + torch.mean(torch.abs(latent_z - z_gen)) + torch.mean(torch.abs(latent_z - z_gen2))
                    layout_rec_loss = torch.mean(torch.abs(layout - layout_dec))
                    object_rec_loss = 0 #torch.mean(torch.abs(torch.cat([norm_mask, object_edge], 1) - object_dec))

                    criterionGAN = nn.BCELoss().cuda()
                    true_tensor = Variable(torch.cuda.FloatTensor(d_theta_gt.data.size()).fill_(1.))
                    fake_tensor = Variable(torch.cuda.FloatTensor(d_theta_gt.data.size()).fill_(0.))
                    d_stn_loss = criterionGAN(d_theta_gt, true_tensor) + criterionGAN(d_theta_rec, fake_tensor) + \
                                 criterionGAN(d_theta_gen, fake_tensor) + criterionGAN(d_theta_gen_ref, fake_tensor)
                    g_stn_loss = criterionGAN(d_theta_rec, true_tensor) + criterionGAN(d_theta_gen, true_tensor) + criterionGAN(d_theta_gen_ref, true_tensor)

                    d_layout_gt = self.dis_sem(layout_gt, object_mask)
                    d_layout_gen = self.dis_sem(layout_gen, object_mask_gen)
                    d_layout_ref_gen = self.dis_sem(layout_ref_gen, obj_mask_ref_gen)
                    true_tensor = Variable(torch.cuda.FloatTensor(d_layout_gt.data.size()).fill_(1.))
                    fake_tensor = Variable(torch.cuda.FloatTensor(d_layout_gt.data.size()).fill_(0.))

                    d_where_loss = criterionGAN(d_layout_gt, true_tensor) + criterionGAN(d_layout_gen, fake_tensor) + criterionGAN(d_layout_ref_gen, fake_tensor)
                    g_where_loss = criterionGAN(d_layout_gen, true_tensor) + criterionGAN(d_layout_ref_gen, true_tensor)

                    if(flag==0): 
                        loss = self.rec_weight*(theta_rec_loss + 0.01*kl_loss) + self.latent_rec_weight*latent_rec_loss + 0*layout_rec_loss + 0*object_rec_loss
                    elif(flag==1):
                        loss = self.rec_weight*(theta_rec_loss + 0.01*kl_loss)+ self.latent_rec_weight*latent_rec_loss + 0*layout_rec_loss + 0*object_rec_loss + self.affine_weight*g_stn_loss + self.layout_weight*g_where_loss
                    elif(flag==2):
                        loss = self.affine_weight*d_stn_loss + self.layout_weight*d_where_loss

                    loss.backward()
                    if(flag==0): 
                        self.E_optim.step()
                    elif(flag==1):
                        self.G_optim.step()
                    elif(flag==2):
                        self.D_optim.step()

                    if(step%50==0):
                        writer.add_scalar("Loss/Rec_theta_loss", theta_rec_loss, step)
                        writer.add_scalar("Loss/Rec_latent_loss", latent_rec_loss, step)
                        writer.add_scalar("Loss/Rec_layout_loss", layout_rec_loss, step)
                        writer.add_scalar("Loss/Rec_object_loss", object_rec_loss, step)
                        writer.add_scalar("Loss/D_STN_loss", d_stn_loss, step)
                        writer.add_scalar("Loss/G_STN_loss", g_stn_loss, step)
                        writer.add_scalar("Loss/D_where_loss", d_where_loss, step)
                        writer.add_scalar("Loss/G_where_loss", g_where_loss, step)
                        writer.add_scalar("Loss/KL_loss", kl_loss)

                    #############
                    if flag ==2: 
                        theta_rec[0, 0, 0] = theta_rec[0, 0, 0] * (local_w/self.patch_w)
                        theta_rec[0, 1, 1] = theta_rec[0, 1, 1] * (local_h/self.patch_h)
                        theta_rec[0, 0, 2] = theta_rec[0, 0, 2] * (local_w/self.patch_w)
                        theta_rec[0, 1, 2] = theta_rec[0, 1, 2] * (local_h/self.patch_h)
                        theta_gen[0, 0, 0] = theta_gen[0, 0, 0] * (local_w/self.patch_w)
                        theta_gen[0, 1, 1] = theta_gen[0, 1, 1] * (local_h/self.patch_h)
                        theta_gen[0, 0, 2] = theta_gen[0, 0, 2] * (local_w/self.patch_w)
                        theta_gen[0, 1, 2] = theta_gen[0, 1, 2] * (local_h/self.patch_h)
                        self.stnet.object_compose(patch_obj, patch_mask, theta_rec, back_image, 'rec', self.result_dir, indx, is_train=True)
                        self.stnet.object_compose(patch_obj, patch_mask, theta_gen, back_image, 'gen', self.result_dir, indx, is_train=True)
                    ###########

                    print("[%5d/%5d] time: %4.4f theta_rec_loss: %.8f, latent_rec_loss: %.8f, layout_rec_loss: %.8f, object_rec_loss: %.8f, d_stn_loss: %.8f, g_stn_loss: %.8f, d_where_loss: %.8f, g_where_loss: %.8f, kl_loss: %.8f" \
                        % (step, self.iteration, time.time() - start_time, theta_rec_loss, latent_rec_loss, layout_rec_loss, object_rec_loss, d_stn_loss, g_stn_loss, d_where_loss, g_where_loss, kl_loss))

                    if step % self.save_freq == 0:
                        self.save(os.path.join(self.result_dir, 'model'), step)

                    if step % 100 == 0:
                        params = {}
                        params['stnet'] = self.stnet.state_dict()
                        params['enc_t'] = self.enc_t.state_dict()
                        params['dis_t'] = self.dis_t.state_dict()
                        params['dis_sem'] = self.dis_sem.state_dict()
                        torch.save(params, os.path.join(self.result_dir, 'params_latest.pt'))

                    step += 1

        writer.flush()

    def save(self, dir, step):
        params = {}
        params['stnet'] = self.stnet.state_dict()
        params['enc_t'] = self.enc_t.state_dict()
        params['dis_t'] = self.dis_t.state_dict()
        params['dis_sem'] = self.dis_sem.state_dict()
        torch.save(params, os.path.join(dir, 'params_%07d.pt' % step))

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, 'params_%07d.pt' % step))
        self.stnet.load_state_dict(params['stnet'])
        self.enc_t.load_state_dict(params['enc_t'])
        self.dis_t.load_state_dict(params['dis_t'])
        self.dis_sem.load_state_dict(params['dis_sem'])


    def object_extract(self, mask, obj, indx):
        mask = mask.numpy()
        img_h = self.img_h
        img_w = self.img_w

        for i in range(img_h):
            line = mask[:, i, :]
            if(np.sum(line) > 0):
                h_1 = i
                break
        for i in range(img_h-1, 0, -1):
            line = mask[:, i, :]
            if(np.sum(line) > 0):
                h_2 = i
                break
        for i in range(img_w):
            column = mask[:, :, i]
            if(np.sum(column) > 0):
                w_1 = i
                break
        for i in range(img_w-1, 0, -1):
            column = mask[:, :, i]
            if(np.sum(column) > 0):
                w_2 = i
                break

        patch_mask = mask[:, h_1:h_2+1, w_1:w_2+1]
        patch_obj = obj[:, h_1:h_2+1, w_1:w_2+1]
        _, obj_h, obj_w = np.shape(patch_mask)
 
        if(obj_h>obj_w): 
            empty_mat_left = np.zeros((obj_h, int((obj_h-obj_w)/2)))
            empty_mat_right = np.zeros((obj_h, obj_h-obj_w-int((obj_h-obj_w)/2)))
            empty_mat_left = np.expand_dims(empty_mat_left, axis=0)
            empty_mat_right = np.expand_dims(empty_mat_right, axis=0)
            empty_mat_obj_left = np.tile(empty_mat_left, (3, 1, 1))
            empty_mat_obj_right = np.tile(empty_mat_right, (3, 1, 1))
            patch_mask = np.concatenate((empty_mat_left, patch_mask, empty_mat_right), axis=2)
            patch_obj = np.concatenate((empty_mat_obj_left, patch_obj, empty_mat_obj_right), axis=2)
            obj_w = obj_h
            w_2 = w_2 + (obj_h-obj_w)
            w = img_w + (obj_h-obj_w)
        else:
            empty_mat_up = np.zeros((int((obj_w-obj_h)/2), obj_w))
            empty_mat_down = np.zeros((obj_w-obj_h-int((obj_w-obj_h)/2), obj_w))
            empty_mat_up = np.expand_dims(empty_mat_up, axis=0)
            empty_mat_down = np.expand_dims(empty_mat_down, axis=0)
            empty_mat_obj_up = np.tile(empty_mat_up, (3, 1, 1))
            empty_mat_obj_down = np.tile(empty_mat_down, (3, 1, 1))
            patch_mask = np.concatenate((empty_mat_up, patch_mask, empty_mat_down), axis=1)
            patch_obj = np.concatenate((empty_mat_obj_up, patch_obj, empty_mat_obj_down), axis=1)
            obj_h = obj_w
            h_1 = h_1-(obj_w-obj_h)
            h = img_h + (obj_w-obj_h)

        # image resize
        norm_mask = np.moveaxis(patch_mask, 0, -1)
        norm_obj = np.moveaxis(patch_obj, 0, -1)
        # cv2.imwrite('results/obj_mask'+indx[n], norm_mask*255)
        # cv2.imwrite('results/oobj'+indx[n], norm_obj*255)

        norm_mask = resize(norm_mask, (self.patch_h, self.patch_w)) # values: [0, 1]
        norm_obj = resize(norm_obj, (self.patch_h, self.patch_w)) # # values: [0, 1]

        # cv2.imwrite('results/res_obj_mask'+indx[n], norm_mask*255)
        # cv2.imwrite('results/res_oobj'+indx[n], norm_obj*255)
        
        norm_mask = np.moveaxis(norm_mask, -1, 0)
        norm_obj = np.moveaxis(norm_obj, -1, 0)

        # scaling parameters
        s_x = self.patch_w*1.0/obj_w
        s_y = self.patch_h*1.0/obj_h

        # translation parameters
        h_o = np.floor((h_1 + h_2)/2)
        w_o = np.floor((w_1 + w_2)/2)
        t_x = (w_o-img_w/2)/(img_w/2)
        t_y = (h_o-img_h/2)/(img_h/2)

        # norm_mask = np.expand_dims(norm_mask, axis=0)
        # norm_obj = np.expand_dims(norm_obj, axis=0)

        theta_gt = [[s_x, 0, -t_x*s_x], 
                    [0, s_y, -t_y*s_y]]
        # theta_gt = np.moveaxis(theta_gt, -1, 0)

        # to tensor
        theta_gt = torch.from_numpy(np.expand_dims(np.float32(theta_gt), axis=0)).to(self.device)
        patch_mask = torch.from_numpy(np.expand_dims(np.float32(patch_mask), axis=0)).to(self.device)
        patch_obj = torch.from_numpy(np.expand_dims(np.float32(patch_obj), axis=0)).to(self.device)
        norm_mask = torch.from_numpy(np.expand_dims(np.float32(norm_mask), axis=0)).to(self.device)

        return (theta_gt, patch_mask, patch_obj, norm_mask, np.float32(norm_obj))


    def edge_extract(self, norm_obj, indx):
        norm_obj = np.moveaxis(norm_obj*255., 0, -1)
        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        norm_obj = rgb2gray(norm_obj)
        edge_detector = CannyEdgeDetector(norm_obj, dimension=1, sigma=2)
        object_edge = edge_detector.detect_edges()
        object_edge = np.expand_dims(object_edge, axis=[0,1])
        object_edge = torch.from_numpy(object_edge/255.).float().to(self.device)
        util.save_img(object_edge[0], self.result_dir, 'train_edge', indx)
        return object_edge


    def re_param(self, mu, logvar, mode): 
        if mode == 'train':
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


    def pad_to_nclass(self, x):
        pad_small = Variable(torch.zeros(1, self.class_num - 1, self.img_h, self.img_w)).cuda()
        pad_before_small = Variable(torch.zeros(1, self.target_class, self.img_h, self.img_w)).cuda()
        pad_after_small = Variable(torch.zeros(1, self.class_num - self.target_class - 1, self.img_h, self.img_w)).cuda()

        if self.target_class == 0:
            padded = torch.cat((x, pad_small), 1)
        elif self.target_class == (self.class_num - 1):
            padded = torch.cat((pad_small, x), 1)
        else:
            padded = torch.cat((pad_before_small, x, pad_after_small), 1)

        return padded



    def test(self):
        model_list = glob(os.path.join(self.result_dir, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            print(iter)
            print(os.path.join(self.result_dir, 'model'))
            self.load(os.path.join(self.result_dir, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        with torch.no_grad():
            self.stnet.eval()

            for object_image, object_mask, back_image, layout, object_mask_ref, layout_ref, indx in self.test_data:

                print(indx)
                ## preprocessing ##
                # extract object patch, mask and theta_gt
                theta_gt, patch_mask, patch_obj, norm_mask, norm_obj = self.object_extract(object_mask, object_image, indx)
                theta_gt_ref, _, _, _, _ = self.object_extract(object_mask_ref, object_image, indx)
                # edge extraction
                object_edge = self.edge_extract(norm_obj, indx)

                ## To cuda ##
                # raw patch size
                _, _, local_h, local_w = np.shape(patch_mask)
                object_mask = torch.unsqueeze(object_mask, dim=0).to(self.device)
                layout = torch.unsqueeze(layout, dim=0).to(self.device)
                back_image = torch.unsqueeze(back_image, dim=0).to(self.device)
                layout_ref = torch.unsqueeze(layout_ref, dim=0).to(self.device)
                patch_obj = patch_obj.to(self.device)
                patch_mask = patch_mask.to(self.device)


                ### INFERENCE
                # random z from Gaussian distribution
                latent_z = torch.FloatTensor(1, 4, 1, 1).normal_(0, 1)
                latent_z = Variable(latent_z).to(self.device)
                latent_z_expand = latent_z.expand([-1, -1, self.img_h, self.img_w])
                layout_n_z_gen = torch.cat([layout, latent_z_expand], 1)
                theta_gen, _, object_mask_gen, z_gen, layout_gen, object_gen = self.stnet(layout_n_z_gen, norm_mask, object_edge, self.result_dir, indx)

                theta_mu, theta_logvar = self.enc_t(theta_gt)
                theta_reparam = self.re_param(theta_mu, theta_logvar, mode='train')
                theta_reparam_expand = theta_reparam.expand([-1, -1, self.img_h, self.img_w])
                layout_n_z_rec = torch.cat([layout, theta_reparam_expand], 1)
                theta_rec, _, object_mask_rec, z_rec, layout_rec, object_rec = self.stnet(layout_n_z_rec, norm_mask, object_edge, self.result_dir, indx)

                # obtain layout
                layout_gt = layout
                layout_gen = layout * (1. - object_mask_gen) + self.pad_to_nclass(object_mask_gen)
                layout_rec = layout * (1. - object_mask_rec) + self.pad_to_nclass(object_mask_rec)
                
                # save layout
                util.save_layout(layout_gt, self.result_dir, 'test_original_layout', indx)
                # util.save_layout(layout_ref_gen, self.result_dir, 'test_fake_layout', indx)
                util.save_layout(layout_gen, self.result_dir, 'test_generated_layout', indx)
                util.save_layout(layout_rec, self.result_dir, 'test_reconstructed_layout', indx)

                #############
                theta_rec[0, 0, 0] = theta_rec[0, 0, 0] * (local_w/self.patch_w)
                theta_rec[0, 1, 1] = theta_rec[0, 1, 1] * (local_h/self.patch_h)
                theta_rec[0, 0, 2] = theta_rec[0, 0, 2] * (local_w/self.patch_w)
                theta_rec[0, 1, 2] = theta_rec[0, 1, 2] * (local_h/self.patch_h)
                self.stnet.object_compose(patch_obj, patch_mask, theta_rec, back_image, 'rec', self.result_dir, indx, is_train=False)

                theta_gen[0, 0, 0] = theta_gen[0, 0, 0] * (local_w/self.patch_w)
                theta_gen[0, 1, 1] = theta_gen[0, 1, 1] * (local_h/self.patch_h)
                theta_gen[0, 0, 2] = theta_gen[0, 0, 2] * (local_w/self.patch_w)
                theta_gen[0, 1, 2] = theta_gen[0, 1, 2] * (local_h/self.patch_h)
                self.stnet.object_compose(patch_obj, patch_mask, theta_gen, back_image, 'gen', self.result_dir, indx, is_train=False)
