from CartoonGAN import CartoonGAN
import argparse
from utils import *

"""parsing and configuration"""

def parse_args():
    desc = "Tensorflow implementation of CartoonGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='face2anime', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--init_epoch', type=int, default=1, help='The number of epochs for weight initialization')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
    parser.add_argument('--decay_flag', type=str2bool, default=False, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')
    parser.add_argument('--adv_weight', type=float, default=1.0, help='Weight about GAN')
    parser.add_argument('--vgg_weight', type=float, default=10.0, help='Weight about VGG19')
    parser.add_argument('--gan_type', type=str, default='gan', help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=8, help='The number of resblock')

    parser.add_argument('--n_dis', type=int, default=3, help='The number of discriminator layer')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')
    parser.add_argument('--sn', type=str2bool, default=False, help='using spectral norm')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=False, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = CartoonGAN(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train' :
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

if __name__ == '__main__':
    main()
