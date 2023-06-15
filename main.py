import argparse
import os
from model import SAC_GAN

from utils.misc import check_folder, str2bool


"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of SAC-GAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_path', help='dataset_path')
    parser.add_argument('--dataset_name', type=str, default='cityscapes', help='dataset_name')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')

    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch size')
    parser.add_argument('--epoch', type=int, default=50, help='The number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--img_h', type=int, default=540, help='The height of image')
    parser.add_argument('--img_w', type=int, default=960, help='The width of image')
    parser.add_argument('--patch_s', type=int, default=64, help='The height/width of object patch')
    parser.add_argument('--class_num', type=int, default=19, help='The number of semantic classes')
    parser.add_argument('--target_class', type=str, default='car', help='car / truck / bus / person')
    parser.add_argument('--is_layout_real', type=str2bool, default=True, help='[Ground truth semantic layout / Predicted semantic layout]')

    parser.add_argument('--layout_dim', type=int, default=128, help='The layout code dimension')
    parser.add_argument('--object_dim', type=int, default=30, help='The object code dimension')

    parser.add_argument('--theta_rec_weight', type=float, default=100, help='Weight for 2D transformation matrix reconstruction loss')
    parser.add_argument('--affine_weight', type=float, default=1, help='Weight for affine loss')
    parser.add_argument('--layout_weight', type=float, default=1, help='Weight for layout loss')
    
    parser.add_argument('--save_freq', type=int, default=100, help='The number of model save freq')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    check_folder(os.path.join(args.result_dir, 'ckpt'))
    check_folder(os.path.join(args.result_dir, 'test', 'image'))
    check_folder(os.path.join(args.result_dir, 'test', 'GT_layout'))
    check_folder(os.path.join(args.result_dir, 'test', 'layout'))
    check_folder(os.path.join(args.result_dir, 'test', 'object_image'))
    check_folder(os.path.join(args.result_dir, 'test', 'mask_image'))
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    engine = SAC_GAN(args)

    # build graph
    engine.build_model(args)

    if args.phase == 'train' :
        engine.train()
        print(" [*] Training finished!")

    if args.phase == 'test' :
        engine.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
