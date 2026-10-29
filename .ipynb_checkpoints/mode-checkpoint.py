import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models 
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as utils


from dataset import *
from dataset_dncnn import Dataset
from utils_dncnn import *

from srgan_model import Generator, Discriminator
from DNCNN import DnCNN
from dncnn_srgan import DNCNN_SRGAN

from losses import  PerceptualLossChexNet
from chexNet import CheXNetFeatureExtractor

from PIL import Image
from skimage.color import rgb2ycbcr

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import glob
import csv

#====================DNCNN TRAINING BLOCK ==============================
def train_rldcnn(args):
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=args.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(channels=1, num_of_layers=args.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # training
    writer = SummaryWriter(args.outf)
    step = 0
    noiseL_B=[5,15] 
    fixed_val = torch.stack([dataset_val[i] for i in range(8)], 0).cuda()
    for epoch in range(args.epochs):
        if epoch < args.milestone:
            current_lr = args.lr
        else:
            current_lr = args.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data

            noise = torch.zeros(img_train.size())
            stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
            for n in range(noise.size()[0]):
                sizeN = noise[0,:,:,:].size()
                noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            ssim_train = batch_SSIM(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f SSIM_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train, ssim_train))

            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                writer.add_scalar('SSIM on training data', ssim_train, step)
            step += 1
        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0
        ssim_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=args.val_noiseL/255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
            out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
            ssim_val += batch_SSIM(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        ssim_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f SSIM_val: %.4f" % (epoch+1, psnr_val, ssim_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('SSIM on validation data', ssim_val, epoch)
     
        with torch.no_grad():
            noise = torch.FloatTensor(fixed_val.size()).uniform_(-1, 1).cuda()
            noise = noise * (args.val_noiseL / 255.)

            imgn = fixed_val + noise
            out = torch.clamp(imgn - model(imgn), 0., 1.)

            Img = utils.make_grid(fixed_val.cpu(), nrow=8, normalize=True, scale_each=True)
            Imgn = utils.make_grid(imgn.cpu(), nrow=8, normalize=True, scale_each=True)
            Irecon = utils.make_grid(out.cpu(), nrow=8, normalize=True, scale_each=True)

            writer.add_image('clean image', Img, epoch)
            writer.add_image('noisy image', Imgn, epoch)
            writer.add_image('reconstructed image', Irecon, epoch)

        # save model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'psnr_val': psnr_val,
            'ssim_val': ssim_val,
            'loss': loss.item()
        }, os.path.join(args.outf, f'checkpoint_{epoch+1:03d}.pth'))

 #====================DNCNN TESTING BLOCK ==============================  
def normalize_test_dncnn(data):
    return data/255.

def test_dncnn(args) :
    results = []

    print('Loading data info ...\n')
    valid_ext = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    files_source = []
    for ext in valid_ext:
         files_source.extend(glob.glob(os.path.join('custom_dataset/DATA_DNCNN',args.test_data, ext)))
    files_source.sort()

    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=args.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    checkpoint_files = sorted(glob.glob(os.path.join(args.outf, "checkpoint_*.pth")))

    for ckpt in checkpoint_files:

        print("="*60)
        print("Testing :", os.path.basename(ckpt))

        checkpoint = torch.load(ckpt, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        psnr_test = 0
        ssim_test = 0

        for f in files_source:

            Img = cv2.imread(f)
            Img = normalize_test_dncnn(np.float32(Img[:, :, 0]))
            Img = np.expand_dims(Img, 0)
            Img = np.expand_dims(Img, 1)

            ISource = torch.Tensor(Img)

            noise = torch.FloatTensor(ISource.size()).normal_(
                mean=0,
                std=args.test_noiseL / 255.
            )

            INoisy = ISource + noise

            ISource = ISource.cuda()
            INoisy = INoisy.cuda()

            with torch.no_grad():
                Out = torch.clamp(INoisy - model(INoisy), 0., 1.)

            

            psnr_test += batch_PSNR(Out, ISource, 1.)
            ssim_test += batch_SSIM(Out, ISource, 1.)

        psnr_test /= len(files_source)
        ssim_test /= len(files_source)

        print("Name Of Checkpoint : %s" % os.path.basename(ckpt))
        print("Average PSNR : %.4f dB" % psnr_test)
        print("Average SSIM : %.4f " % ssim_test)

        results.append([
            os.path.basename(ckpt),
            psnr_test,
            ssim_test
        ])

    with open("dncnn_test_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Checkpoint", "PSNR", "SSIM"])
        writer.writerows(results)

 #====================DNCNN INFERENCE BLOCK ============================== 
def test_only_dncnn(args) :

    # ======================
    # CONFIG
    # ======================
    checkpoint_dir = args.outf
    image_path = args.test_denoising_image
    save_root = "DNCNN_inference_results"
    os.makedirs(save_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = img.astype(np.float32) / 255.
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img).to(device)

    # --------------------------
    # Blind Noise (dibuat SEKALI)
    # --------------------------
    sigma = np.random.uniform(5,15)

    noise = torch.randn_like(img) * (sigma/255.)
    noisy = torch.clamp(img + noise,0.,1.)

    print(f"Noise sigma : {sigma:.2f}")

    # Model
    model = DnCNN(channels=1, num_of_layers=17)
    model = nn.DataParallel(model).to(device)

    checkpoint_files = sorted(
        glob.glob(os.path.join(checkpoint_dir,"checkpoint_*.pth"))
    )

    for ckpt in checkpoint_files:

        print("Testing :", os.path.basename(ckpt))

        checkpoint = torch.load(
            ckpt,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():

            predicted_noise = model(noisy)

            denoised = torch.clamp(
                noisy - predicted_noise,
                0.,
                1.
            )


        # Save
        save_dir = os.path.join(
            save_root,
            os.path.splitext(os.path.basename(ckpt))[0]
        )

        os.makedirs(save_dir, exist_ok=True)

        clean = (img.squeeze().cpu().numpy()*255).astype(np.uint8)
        noisy_img = (noisy.squeeze().cpu().numpy()*255).astype(np.uint8)
        denoised_img = (denoised.squeeze().cpu().numpy()*255).astype(np.uint8)

        cv2.imwrite(os.path.join(save_dir,"clean.png"),clean)
        cv2.imwrite(os.path.join(save_dir,"noisy.png"),noisy_img)
        cv2.imwrite(os.path.join(save_dir,"denoised.png"),denoised_img)


 #====================SRGAN TRAINING BLOCK ==============================  
 
def calculate_psnr(gt, output):
    mse = torch.mean((gt - output) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

def calculate_ssim(gt, output):
    gt_np = gt.cpu().numpy()  # Konversi ke NumPy
    output_np = output.cpu().numpy()

    # Pastikan ukuran gambar lebih besar dari win_size (default 7)
    min_dim = min(gt_np.shape[0], gt_np.shape[1])  # Ambil dimensi terkecil
    win_size = 7 if min_dim >= 7 else min_dim  # Atur win_size agar tidak lebih besar dari gambar

    return ssim(gt_np, output_np, data_range=gt_np.max() - gt_np.min(), win_size=win_size, channel_axis=-1)

def fix_key(key):
    key = re.sub(r'(\D)\.(\d)', r'\1\2', key)
    return key

def train_srgan(args):

    g_loss_list = []
    d_loss_list = []
    ssim_list = []
    psnr_list = []
    epochs_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([crop_srgan(args.scale, args.patch_size), augmentation_srgan()])
    dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=args.in_memory, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num, scale=args.scale)
    
    if args.fine_tuning:
        generator.load_state_dict(torch.load(args.generator_path))
    
    generator = generator.to(device)
    generator.train()

    discriminator = Discriminator(patch_size=args.patch_size * args.scale).to(device)
    discriminator.train()
    
    l2_loss = nn.MSELoss()
    cross_ent = nn.BCELoss()


    g_optim = optim.Adam(generator.parameters(), lr=1e-4)
    d_optim = optim.Adam(discriminator.parameters(), lr=1e-4) 
    scheduler = optim.lr_scheduler.StepLR(g_optim, step_size=100, gamma=0.1)
    
    #CheXNet
    # Load checkpoint dulu
    checkpoint = torch.load('pretrained_models/chexnet_model.pth.tar')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith('module.'):
            new_key = k[7:]  # hapus 'module.'
        if new_key.startswith('densenet121.'):
            new_key = new_key[len('densenet121.'):]  # hapus 'densenet121.'
        if new_key.startswith('classifier.0.'):
            new_key = new_key.replace('classifier.0.', 'classifier.')
        new_key = fix_key(new_key)

        new_state_dict[new_key] = v

    print(list(new_state_dict.keys())[:10])
    chexnet = models.densenet121(pretrained=False)
    chexnet.classifier = nn.Linear(1024, 14)
    chexnet.load_state_dict(new_state_dict)
    chexnet.eval().to(device)

    feature_extractor = CheXNetFeatureExtractor(chexnet, selected_layers=['denseblock2', 'denseblock3']).to(device)
    perceptualloss_ChexNet = PerceptualLossChexNet(feature_extractor).to(device)


    real_label = torch.ones((args.batch_size, 1)).to(device)
    fake_label = torch.zeros((args.batch_size, 1)).to(device)

    fine_epoch = 0

    while fine_epoch < args.fine_train_epoch:
        
        scheduler.step()
        
        for i, tr_data in enumerate(loader):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)

            ## Training Discriminator
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)

            d_loss_real = cross_ent(real_prob, real_label)
            d_loss_fake = cross_ent(fake_prob, fake_label)

            d_loss = d_loss_real + d_loss_fake

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            ## Training Generator
            output, _ = generator(lr)
            fake_prob = discriminator(output)

            _percep_loss = perceptualloss_ChexNet(gt,output)
      

            L2_loss = l2_loss(output, gt)
            percep_loss = args.chexnet_coeff * _percep_loss
            adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label)
            g_loss = percep_loss + adversarial_loss + L2_loss
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

      
        fine_epoch += 1

        # Hitung SSIM dan PSNR setiap 2 epoch
        if fine_epoch % 1 == 0:
            with torch.no_grad():
                ssim_value = calculate_ssim(gt, output)
                psnr_value = calculate_psnr(gt, output)

            print(f"Epoch {fine_epoch}")
            print(f"G Loss: {g_loss.item():.4f} | D Loss: {d_loss.item():.4f}")
            print(f"SSIM: {ssim_value:.4f} | PSNR: {psnr_value:.2f} dB")
            print('=========')

            # Simpan hasil ke list
            g_loss_list.append(g_loss.item())
            d_loss_list.append(d_loss.item())
            ssim_list.append(ssim_value)
            psnr_list.append(psnr_value)
            epochs_list.append(fine_epoch)

        # Simpan model setiap 25 epoch
        if fine_epoch % 1 == 0:
            torch.save(generator.state_dict(), f'./model/SRGAN_gene_{fine_epoch:03d}.pt')
            torch.save(discriminator.state_dict(), f'./model/SRGAN_discrim_{fine_epoch:03d}.pt')

    # Plot hasil training
    ubah_ke_csv(epochs_list, g_loss_list, d_loss_list, ssim_list, psnr_list)

def ubah_ke_csv(epochs, g_loss, d_loss, ssim, psnr, filename="hasil_training.csv"):
    def to_list(tensor):
        if isinstance(tensor, torch.Tensor):  # Cek apakah tensor
            return tensor.detach().cpu().tolist()  # Pindah ke CPU, detach, lalu ke list
        elif isinstance(tensor, list):  # Jika sudah list, konversi setiap elemennya
            return [to_list(t) for t in tensor]  # Rekursif jika masih ada tensor di dalam list
        return tensor  # Jika bukan tensor, biarkan apa adanya

    # Pastikan semua data dalam bentuk list biasa
    epochs = to_list(epochs)
    g_loss = to_list(g_loss)
    d_loss = to_list(d_loss)
    ssim = to_list(ssim)
    psnr = to_list(psnr)

    df = pd.DataFrame(
        {
        "Epochs": epochs,
        "Generator Loss": g_loss,
        "Discriminator Loss": d_loss,
        "SSIM": ssim,
        "PSNR": psnr
    }
    )

    df.to_csv(filename, index=False)
    
 #====================SRGAN TESTING BLOCK ==============================  

# In[ ]:
def test(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = mydata(
        GT_path=args.GT_path,
        LR_path=args.LR_path,
        in_memory=False,
        transform=None
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    generator = Generator(
            img_feat=3,
            n_feats=64,
            kernel_size=3,
            num_block=args.res_num
        ).to(device)

    model_files = sorted(
        glob.glob(
            os.path.join(
                args.generator_model_dir_path,
                "SRGAN_gene_*.pt"
            )
        )
    )

    with open("./result.txt", "w") as f:

        with torch.no_grad():

            for model_name in model_files:

                psnr_list = []
                ssim_list = []

                print(f"Testing : {model_name}")

                generator.load_state_dict(
                    torch.load(
                        model_name,
                        map_location=device
                    )
                )

                generator.eval()

                for te_data in loader:

                    gt = te_data['GT'].to(device)
                    lr = te_data['LR'].to(device)

                    bs, c, h, w = lr.size()

                    gt = gt[:, :, :h*args.scale, :w*args.scale]

                    output,_ = generator(lr)

                    output = output[0].cpu().numpy()
                    gt = gt[0].cpu().numpy()

                    output = np.clip(output, -1.0, 1.0)

                    output = (output + 1.0) / 2.0
                    gt = (gt + 1.0) / 2.0



                    output = output.transpose(1, 2, 0)
                    gt = gt.transpose(1, 2, 0)

                    y_output = rgb2ycbcr(output)[
                        args.scale:-args.scale,
                        args.scale:-args.scale,
                        :1
                    ]

                    y_gt = rgb2ycbcr(gt)[
                        args.scale:-args.scale,
                        args.scale:-args.scale,
                        :1
                    ]

                    psnr = compare_psnr(
                        y_output / 255.0,
                        y_gt / 255.0,
                        data_range=1.0
                    )

                    ssim_val = ssim(
                        y_output / 255.0,
                        y_gt / 255.0,
                        data_range=1.0,
                        win_size=7,
                        channel_axis=-1
                    )

                    psnr_list.append(psnr)
                    ssim_list.append(ssim_val)

                avg_psnr = np.mean(psnr_list)
                avg_ssim = np.mean(ssim_list)

                print(
                    f"PSNR: {avg_psnr:.4f} | "
                    f"SSIM: {avg_ssim:.4f}"
                )

                f.write(f"Model : {os.path.basename(model_name)}\n")
                f.write(f"PSNR  : {avg_psnr:.4f}\n")
                f.write(f"SSIM  : {avg_ssim:.4f}\n\n")


 #====================SRGAN INFERENCE BLOCK ==============================  
def test_only_dsasrgan(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = testOnly_data_dsa_srgan(
        LR_path=args.LR_path,
        in_memory=False,
        transform=None
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    model_paths = sorted(
        glob.glob(os.path.join(args.generator_model_dir_path,
                               "DSA_SRGAN_gene_*.pt"))
    )

    if len(model_paths) == 0:
        raise ValueError("Tidak ada model DSA_SRGAN ditemukan!")

    os.makedirs("./result", exist_ok=True)

    for model_path in model_paths:

        model_name = os.path.splitext(os.path.basename(model_path))[0]

        print(f"Testing model: {model_name}")

        # ================= Model ===================
        model = DNCNN_SRGAN(
            channels=1,
            num_of_layers=17,
            img_feat=1,
            n_feats=64,
            num_block=args.res_num,
            scale=args.scale
        ).to(device)

        # Load DnCNN pretrained
        checkpoint = torch.load(args.dncnn_pretrained, map_location=device,  weights_only=False)

        new_state_dict = {}

        for k, v in checkpoint["model_state_dict"].items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v

        if "model_state_dict" in checkpoint:
            model.denoising_module.load_state_dict(new_state_dict)
        else:
            model.denoising_module.load_state_dict(new_state_dict)

        # Load Generator hasil training DSA-SRGAN
        model.generator.load_state_dict(torch.load(model_path, map_location=device))

        model.eval()

        # ================= Testing =================
        with torch.no_grad():

            for i, te_data in enumerate(loader):

                lr = te_data["LR"].to(device)

                output, _ = model(lr)

                output = output.squeeze(0).cpu().numpy()

                output = (output + 1.0) / 2.0
                output = np.clip(output, 0, 1)

                # (1,H,W) -> (H,W)
                output = output.squeeze(0)

                result = Image.fromarray((output * 255).astype(np.uint8))

                result.save(f"./result/{model_name}_res_{i:04d}.png")


def test_only_srgan(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = testOnly_data(
        LR_path=args.LR_path,
        in_memory=False,
        transform=None
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )


    model_paths = sorted(glob.glob(os.path.join(args.generator_model_dir_path, "SRGAN_gene_*.pt")))
    


    if len(model_paths) == 0:
        raise ValueError("Tidak ada SRGAN_gene model ditemukan!")

    os.makedirs("./result", exist_ok=True)

    for model_path in model_paths:

        model_name = os.path.splitext(os.path.basename(model_path))[0]

        print(f"Testing model: {model_name}")

        generator = Generator(
            img_feat=3,
            n_feats=64,
            kernel_size=3,
            num_block=args.res_num
        )

        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator = generator.to(device)
        generator.eval()

        with torch.no_grad():
            for i, te_data in enumerate(loader):
                lr = te_data['LR'].to(device)

                output, _ = generator(lr)
                output = output[0].cpu().numpy()
                output = (output + 1.0) / 2.0
                output = output.transpose(1, 2, 0)

                result = Image.fromarray((output * 255.0).astype(np.uint8))

  
                result.save(f"./result/{model_name}_res_{i:04d}.png")



 #====================DNCNN SRGAN(DSA_SRGAN) JOINT MODEL TRAINING ==============================  

def train_dncnn_srgan(args):
    DSA_SRGAN_g_loss_list = []
    DSA_SRGAN_d_loss_list = []
    DSA_SRGAN_ssim_list = []
    DSA_SRGAN_psnr_list = []
    DSA_SRGAN_epochs_list = []

    # parameter
    noiseL_B = [5, 15] 
    fine_epoch = 0

    # Load pretrained DnCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dsa_srgan = DNCNN_SRGAN(
                        channels=1,
                        num_of_layers=17,
                        img_feat=1,
                        n_feats=64,
                        num_block=args.res_num,
                        scale=args.scale
                    ).to(device)
  
    checkpoint = torch.load(args.dncnn_pretrained, map_location=device, weights_only=False)
    new_state_dict = {}
    for k, v in checkpoint["model_state_dict"].items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    model_dsa_srgan.denoising_module.load_state_dict(new_state_dict)

    # Freeze DnCNN
    for param in model_dsa_srgan.denoising_module.parameters():
        param.requires_grad = False

    transform = transforms.Compose([crop_srgan(args.scale, args.patch_size), augmentation_srgan()])
    dataset = mydata_dsa_srgan(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=args.in_memory, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    #===========================SRGAN===============================================
    if args.fine_tuning:
        model_dsa_srgan.generator.load_state_dict(
            torch.load(args.generator_path, map_location=device)
        )  

    model_dsa_srgan.train()
    model_dsa_srgan.denoising_module.eval() # Pastikan DnCNN tetap mode eval (freeze BN/Dropout)
    model_dsa_srgan.generator.apply(weights_init_kaiming)
    
    discriminator = Discriminator(patch_size=args.patch_size * args.scale).to(device)
    discriminator.apply(weights_init_kaiming)
    discriminator.train()
    
    l2_loss = nn.MSELoss()
    cross_ent = nn.BCELoss()

    g_optim = optim.Adam(model_dsa_srgan.generator.parameters(), lr=1e-4)
    d_optim = optim.Adam(discriminator.parameters(), lr=1e-4) 
    scheduler = optim.lr_scheduler.StepLR(g_optim, step_size=50, gamma=0.1)
    
    #===========================CheXNet===============================================
    checkpoint = torch.load('pretrained_models/chexnet_model.pth.tar')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith('module.'):
            new_key = k[7:]
        if new_key.startswith('densenet121.'):
            new_key = new_key[len('densenet121.'):]
        if new_key.startswith('classifier.0.'):
            new_key = new_key.replace('classifier.0.', 'classifier.')
        new_key = fix_key(new_key)
        new_state_dict[new_key] = v

    chexnet = models.densenet121(pretrained=False)
    chexnet.classifier = nn.Linear(1024, 14)
    chexnet.load_state_dict(new_state_dict)
    chexnet.eval().to(device)
    feature_extractor = CheXNetFeatureExtractor(chexnet, selected_layers=['denseblock2', 'denseblock3']).to(device)
    perceptualloss_ChexNet = PerceptualLossChexNet(feature_extractor).to(device)

    real_label = torch.ones((args.batch_size, 1)).to(device)
    fake_label = torch.zeros((args.batch_size, 1)).to(device)

    #===========================Training===============================================
    while fine_epoch < args.fine_train_epoch: 
        # PENTING: Jangan taruh scheduler.step() di sini
        # TAMBAHKAN ATAU PASTIKAN DI SINI:
        model_dsa_srgan.train()               # Mengatur seluruh pipeline ke mode train
        model_dsa_srgan.denoising_module.eval() # PAKSA ULANG DnCNN agar tetap eval (BN & Dropout terkunci
        for i, tr_data in enumerate(loader):
            gt = tr_data['GT'].to(device)
            lr = tr_data['LR'].to(device)

            # Generate noise
            noise = torch.zeros_like(lr)
            stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=lr.size()[0])
            for n in range(lr.size(0)):
                noise[n] = torch.randn_like(lr[n]) * (stdN[n] / 255.)

            lr_noisy = lr + noise

            # OPTIMASI: Forward pass generator sekali saja untuk satu batch
            output, feat = model_dsa_srgan(lr_noisy)

            #===========================Discriminator===============================================
            # Detach output agar gradien tidak mengalir kembali ke generator saat update discriminator
            fake_prob = discriminator(output.detach())
            real_prob = discriminator(gt)

            d_loss_real = cross_ent(real_prob, real_label)
            d_loss_fake = cross_ent(fake_prob, fake_label)
            d_loss = d_loss_real + d_loss_fake

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            #===========================Generator + DnCNN===============================================
            # Tidak perlu forward pass generator lagi, gunakan `output` yang tadi
            fake_prob_g = discriminator(output)

            _percep_loss = perceptualloss_ChexNet(gt, output)
            L2_loss = l2_loss(output, gt)
            percep_loss = args.chexnet_coeff * _percep_loss
            adversarial_loss = args.adv_coeff * cross_ent(fake_prob_g, real_label)
            
            g_loss = percep_loss + adversarial_loss + L2_loss
            
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

        fine_epoch += 1
        scheduler.step() # PERBAIKAN: Scheduler dipanggil setelah epoch selesai

        # Hitung SSIM dan PSNR setiap epoch (menggunakan batch terakhir)
        if fine_epoch % 1 == 0:
            with torch.no_grad():
                # Pastikan fungsi metrik ini menerima tensor range [0, 1]
                ssim_value = calculate_ssim_dsa_srgan(gt, output)
                psnr_value = calculate_psnr_dsa_srgan(gt, output)

            print(f"Epoch {fine_epoch}")
            print(f"G Loss: {g_loss.item():.4f} | D Loss: {d_loss.item():.4f}")
            print(f"SSIM: {ssim_value:.4f} | PSNR: {psnr_value:.2f} dB")
            print('=========')

            # Simpan hasil ke list
            DSA_SRGAN_g_loss_list.append(g_loss.item())
            DSA_SRGAN_d_loss_list.append(d_loss.item())
            DSA_SRGAN_ssim_list.append(ssim_value)
            DSA_SRGAN_psnr_list.append(psnr_value)
            DSA_SRGAN_epochs_list.append(fine_epoch)

        # Simpan model setiap epoch
        if fine_epoch % 1 == 0:
            torch.save(model_dsa_srgan.generator.state_dict(), f'./model_DSA_SRGAN/DSA_SRGAN_gene_{fine_epoch:03d}.pt')
            torch.save(discriminator.state_dict(), f'./model_DSA_SRGAN/DSA_SRGAN_discrim_{fine_epoch:03d}.pt')

    # Plot/Simpan hasil training ke CSV
    ubah_ke_csv(
        DSA_SRGAN_epochs_list,
        DSA_SRGAN_g_loss_list,
        DSA_SRGAN_d_loss_list,
        DSA_SRGAN_ssim_list,
        DSA_SRGAN_psnr_list
    )


def calculate_psnr_dsa_srgan(gt, output):
    psnr_list = []
    for i in range(gt.size(0)):
        mse = torch.mean((gt[i] - output[i]) ** 2)
        if mse.item() == 0:
            psnr_list.append(100.0)
        else:
            max_pixel = 1.0  # Asumsi gambar berada pada range [0, 1]
            psnr = 20 * torch.log10(torch.tensor(max_pixel, device=mse.device) / torch.sqrt(mse))
            psnr_list.append(psnr.item())
    return np.mean(psnr_list)


def calculate_ssim_dsa_srgan(gt, output):
    gt_np = gt.detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    ssim_list = []

    for i in range(gt_np.shape[0]):
        # Grayscale
        if gt_np.shape[1] == 1:
            gt_img = gt_np[i, 0]
            out_img = output_np[i, 0]
            score = ssim(
                gt_img,
                out_img,
                data_range=1.0 # PERBAIKAN: Di-lock ke nilai 1.0 agar konsisten
            )
        # RGB
        else:
            gt_img = np.transpose(gt_np[i], (1, 2, 0))
            out_img = np.transpose(output_np[i], (1, 2, 0))
            score = ssim(
                gt_img,
                out_img,
                data_range=1.0, # PERBAIKAN: Di-lock ke nilai 1.0 agar konsisten
                channel_axis=-1
            )
        ssim_list.append(score)

    return np.mean(ssim_list)