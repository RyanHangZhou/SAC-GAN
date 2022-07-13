from SAC_GAN import *
import argparse
from util import *

"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of SAC-GAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_path', help='dataset_path')
    parser.add_argument('--dataset_name', type=str, default='cityscapes', help='dataset_name')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')

    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--epoch', type=int, default=50, help='The number of training epochs')
    parser.add_argument('--iteration', type=int, default=20000, help='The number of training iterations')
    parser.add_argument('--lr', type=float, default=2e-4, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--img_h', type=int, default=540, help='The height of image')
    parser.add_argument('--img_w', type=int, default=960, help='The width of image')
    parser.add_argument('--patch_h', type=int, default=64, help='The height of object patch')
    parser.add_argument('--patch_w', type=int, default=64, help='The width of object patch')
    parser.add_argument('--class_num', type=int, default=19, help='The number of semantic classes')
    parser.add_argument('--target_class', type=int, default=13, help='The class number to insert an object patch')
    parser.add_argument('--layout_flag', type=str2bool, default=True, help='[Ground truth semantic layout / Predicted semantic layout]')

    parser.add_argument('--rec_weight', type=float, default=100, help='Weight for reconstruction loss')
    parser.add_argument('--latent_rec_weight', type=float, default=0.05, help='Weight for latent reconstruction loss')
    parser.add_argument('--affine_weight', type=float, default=1, help='Weight for affine loss')
    parser.add_argument('--layout_weight', type=float, default=1, help='Weight for layout loss')
    
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)


    # parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')
    # parser.add_argument('--U2Net', type=str2bool, default=False, help='[U2-Net / U-GAT-IT network]')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, 'model'))
    # check_folder(os.path.join(args.result_dir, 'img'))
    # check_folder(os.path.join(args.result_dir, 'test'))

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = SAC_GAN(args)

    # build graph
    gan.build_model()

    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
