from mode import *
import argparse
from dataset_dncnn import prepare_data

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

#==============GENERAL=================#
parser.add_argument("--res_num", type = int, default = 16)
parser.add_argument("--num_workers", type = int, default = 0)
parser.add_argument("--batch_size", type = int, default = 16)
parser.add_argument("--fine_train_epoch", type = int, default =100)
parser.add_argument("--fine_tuning", type = str2bool, default = False)
parser.add_argument("--in_memory", type = str2bool, default = True)
parser.add_argument("--mode", type = str, default = 'train_srgan')


#==============SRGAN=================#
parser.add_argument("--LR_path", type = str, default = 'custom_dataset/LR_DATASET_ONLY/TRAIN')
parser.add_argument("--GT_path", type = str, default = 'custom_dataset/HR_DATASET/TRAIN')
parser.add_argument("--generator_path", type = str)
parser.add_argument("--generator_model_dir_path", type=str, default="model_DSA_SRGAN/")
parser.add_argument("--L2_coeff", type = float, default = 1.0)
parser.add_argument("--adv_coeff", type = float, default = 0.001)
parser.add_argument("--chexnet_coeff", type = float, default = 0.005)
parser.add_argument("--scale", type = int, default = 4)
parser.add_argument("--patch_size", type = int, default = 32)

#==============DNCNN=================#
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs_dncnn", help='path of log files')
parser.add_argument("--test_data", type=str, default='TEST', help='test set')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--val_noiseL", type=float, default=10, help='noise level used on validation set')
parser.add_argument("--test_denoising_image", type=str, default='test_data/citra_uji_coba.jpg', help='noise level used on validation set')

#==============DSA SRGAN=================#
parser.add_argument("--dncnn_pretrained", type=str, default="logs_dncnn/checkpoint_010.pth", help='pre trained dncnn model')


args = parser.parse_args()

if args.mode == 'train_srgan':
    train_srgan(args)

elif args.mode == 'test_srgan':
    test(args)

elif args.mode == 'test_only_srgan':
    test_only_srgan(args)

elif args.mode == 'train_med_dncnn':
    if args.preprocess:
        prepare_data(data_path='custom_dataset/DATA_DNCNN', patch_size=50, stride=50, aug_times=1)
    train_rldcnn(args)
    
elif args.mode == 'test_med_dncnn':
    test_dncnn(args)

elif args.mode == 'inference_med_dncnn':
    test_only_dncnn(args)

elif args.mode == 'train_dncnn_srgan':
    train_dncnn_srgan(args)


elif args.mode == 'test_only_dncnn_srgan':
    test_only_dsasrgan(args)

