import torch
import sys
import os
from tqdm import tqdm
import math
import torch.nn as nn
import torch.optim as optim
from IPython import embed
import math
import cv2
import string
from PIL import Image
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from collections import OrderedDict

from model import bicubic, srcnn, vdsr, srresnet, edsr, esrgan, rdn, lapsrn, tsrn
from model import recognizer
from model import moran
from model import crnn
from dataset import lmdbDataset, alignCollate_real, ConcatDataset, lmdbDataset_real, alignCollate_syn, lmdbDataset_mix
from loss import gradient_loss, percptual_loss, image_loss

from utils.labelmaps import get_vocabulary, labels2strs

sys.path.append('../')
from utils import util, ssim_psnr, utils_moran, utils_crnn
import dataset.dataset as dataset


class TextBase(object):
    def __init__(self, config, args):
        super(TextBase, self).__init__()
        self.config = config
        self.args = args
        self.scale_factor = self.config.TRAIN.down_sample_scale
        if self.args.syn:
            self.align_collate = alignCollate_syn
            self.load_dataset = lmdbDataset
        elif self.args.mixed:
            self.align_collate = alignCollate_real
            self.load_dataset = lmdbDataset_mix
        else:
            self.align_collate = alignCollate_real
            self.load_dataset = lmdbDataset_real
        self.resume = args.resume if args.resume is not None else config.TRAIN.resume
        self.batch_size = args.batch_size if args.batch_size is not None else self.config.TRAIN.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha_dict = {
            'digit': string.digits,
            'lower': string.digits + string.ascii_lowercase,
            'upper': string.digits + string.ascii_letters,
            'all': string.digits + string.ascii_letters + string.punctuation
        }
        self.test_data_dir = self.args.test_data_dir if self.args.test_data_dir is not None else self.config.TEST.test_data_dir
        self.voc_type = self.config.TRAIN.voc_type
        self.alphabet = alpha_dict[self.voc_type]
        self.max_len = config.TRAIN.max_len
        self.vis_dir = self.args.vis_dir if self.args.vis_dir is not None else self.config.TRAIN.VAL.vis_dir
        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.mask = self.args.mask
        alphabet_moran = ':'.join(string.digits+string.ascii_lowercase+'$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')
        self.converter_crnn = utils_crnn.strLabelConverter(string.digits + string.ascii_lowercase)

    def get_train_data(self):
        cfg = self.config.TRAIN
        if isinstance(cfg.train_data_dir, list):
            dataset_list = []
            for data_dir_ in cfg.train_data_dir:
                dataset_list.append(
                    self.load_dataset(root=data_dir_,
                                      voc_type=cfg.voc_type,
                                      max_len=cfg.max_len))
            train_dataset = dataset.ConcatDataset(dataset_list)
        else:
            raise TypeError('check trainRoot')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=True)
        return train_dataset, train_loader

    def get_val_data(self):
        cfg = self.config.TRAIN
        assert isinstance(cfg.VAL.val_data_dir, list)
        dataset_list = []
        loader_list = []
        for data_dir_ in cfg.VAL.val_data_dir:
            val_dataset, val_loader = self.get_test_data(data_dir_)
            dataset_list.append(val_dataset)
            loader_list.append(val_loader)
        return dataset_list, loader_list

    def get_test_data(self, dir_):
        cfg = self.config.TRAIN
        self.args.test_data_dir
        test_dataset = self.load_dataset(root=dir_,
                                         voc_type=cfg.voc_type,
                                         max_len=cfg.max_len,
                                         test=True,
                                         )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=int(cfg.workers),
            collate_fn=self.align_collate(imgH=cfg.height, imgW=cfg.width, down_sample_scale=cfg.down_sample_scale,
                                          mask=self.mask),
            drop_last=False)
        return test_dataset, test_loader

    def generator_init(self):
        cfg = self.config.TRAIN
        if self.args.arch == 'tsrn':
            model = tsrn.TSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                       STN=self.args.STN, mask=self.mask, srb_nums=self.args.srb, hidden_units=self.args.hd_u)
            image_crit = image_loss.ImageLoss(gradient=self.args.gradient, loss_weight=[1, 1e-4])
        elif self.args.arch == 'bicubic' and self.args.test:
            model = bicubic.BICUBIC(scale_factor=self.scale_factor)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'srcnn':
            model = srcnn.SRCNN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'vdsr':
            model = vdsr.VDSR(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'srres':
            model = srresnet.SRResNet(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height,
                                      STN=self.args.STN, mask=self.mask)
            image_crit = nn.MSELoss()
        elif self.args.arch == 'esrgan':
            model = esrgan.RRDBNet(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'rdn':
            model = rdn.RDN(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'edsr':
            model = edsr.EDSR(scale_factor=self.scale_factor)
            image_crit = nn.L1Loss()
        elif self.args.arch == 'lapsrn':
            model = lapsrn.LapSRN(scale_factor=self.scale_factor, width=cfg.width, height=cfg.height, STN=self.args.STN)
            image_crit = lapsrn.L1_Charbonnier_loss()
        else:
            raise ValueError
        if self.args.arch != 'bicubic':
            model = model.to(self.device)
            image_crit.to(self.device)
            if cfg.ngpu > 1:
                model = torch.nn.DataParallel(model, device_ids=range(cfg.ngpu))
                image_crit = torch.nn.DataParallel(image_crit, device_ids=range(cfg.ngpu))
            if self.resume is not '':
                print('loading pre-trained model from %s ' % self.resume)
                if self.config.TRAIN.ngpu == 1:
                    model.load_state_dict(torch.load(self.resume)['state_dict_G'])
                else:
                    model.load_state_dict(
                        {'module.' + k: v for k, v in torch.load(self.resume)['state_dict_G'].items()})
        return {'model': model, 'crit': image_crit}

    def optimizer_init(self, model):
        cfg = self.config.TRAIN
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr,
                               betas=(cfg.beta1, 0.999))
        return optimizer

    def tripple_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, index):
        for i in (range(self.config.TRAIN.VAL.n_vis)):
            # embed()
            tensor_in = image_in[i][:3,:,:]
            transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                 transforms.ToTensor()]
            )

            tensor_in = transform(tensor_in.cpu())
            tensor_out = image_out[i][:3,:,:]
            tensor_target = image_target[i][:3,:,:]
            images = ([tensor_in, tensor_out.cpu(), tensor_target.cpu()])
            vis_im = torch.stack(images)
            vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
            out_root = os.path.join('./demo', self.vis_dir)
            if not os.path.exists(out_root):
                os.mkdir(out_root)
            out_path = os.path.join(out_root, str(index))
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            im_name = pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
            im_name = im_name.replace('/', '')
            if index is not 0:
                torchvision.utils.save_image(vis_im, os.path.join(out_path, im_name), padding=0)

    def test_display(self, image_in, image_out, image_target, pred_str_lr, pred_str_sr, label_strs, str_filt):
        visualized = 0
        for i in (range(image_in.shape[0])):
            if True:
                if (str_filt(pred_str_lr[i], 'lower') != str_filt(label_strs[i], 'lower')) and \
                        (str_filt(pred_str_sr[i], 'lower') == str_filt(label_strs[i], 'lower')):
                    visualized += 1
                    tensor_in = image_in[i].cpu()
                    tensor_out = image_out[i].cpu()
                    tensor_target = image_target[i].cpu()
                    transform = transforms.Compose(
                        [transforms.ToPILImage(),
                         transforms.Resize((image_target.shape[-2], image_target.shape[-1]), interpolation=Image.BICUBIC),
                         transforms.ToTensor()]
                    )
                    tensor_in = transform(tensor_in)
                    images = ([tensor_in, tensor_out, tensor_target])
                    vis_im = torch.stack(images)
                    vis_im = torchvision.utils.make_grid(vis_im, nrow=1, padding=0)
                    out_root = os.path.join('./display', self.vis_dir)
                    if not os.path.exists(out_root):
                        os.mkdir(out_root)
                    if not os.path.exists(out_root):
                        os.mkdir(out_root)
                    im_name = pred_str_lr[i] + '_' + pred_str_sr[i] + '_' + label_strs[i] + '_.png'
                    im_name = im_name.replace('/', '')
                    torchvision.utils.save_image(vis_im, os.path.join(out_root, im_name), padding=0)
        return visualized

    def save_checkpoint(self, netG, epoch, iters, best_acc_dict, best_model_info, is_best, converge_list):
        ckpt_path = os.path.join('ckpt', self.vis_dir)
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        save_dict = {
            'state_dict_G': netG.module.state_dict(),
            'info': {'arch': self.args.arch, 'iters': iters, 'epochs': epoch, 'batch_size': self.batch_size,
                     'voc_type': self.voc_type, 'up_scale_factor': self.scale_factor},
            'best_history_res': best_acc_dict,
            'best_model_info': best_model_info,
            'param_num': sum([param.nelement() for param in netG.module.parameters()]),
            'converge': converge_list
        }
        if is_best:
            torch.save(save_dict, os.path.join(ckpt_path, 'model_best.pth'))
        else:
            torch.save(save_dict, os.path.join(ckpt_path, 'checkpoint.pth'))

    def MORAN_init(self):
        cfg = self.config.TRAIN
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        model_path = self.config.TRAIN.VAL.moran_pretrained
        print('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)
        MORAN = torch.nn.DataParallel(MORAN, device_ids=range(cfg.ngpu))
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN

    def parse_moran_data(self, imgs_input):
        batch_size = imgs_input.shape[0]
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def CRNN_init(self):
        model = crnn.CRNN(32, 1, 37, 256)
        model = model.to(self.device)
        model_path = self.config.TRAIN.VAL.crnn_pretrained
        print('loading pretrained crnn model from %s' % model_path)
        model.load_state_dict(torch.load(model_path))
        return model

    def parse_crnn_data(self, imgs_input):
        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, 100), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def Aster_init(self):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=aster_info.max_len,
                                             eos=aster_info.char2id[aster_info.EOS], STN_ON=True)
        aster.load_state_dict(torch.load(self.config.TRAIN.VAL.rec_pretrained)['state_dict'])
        print('load pred_trained aster model from %s' % self.config.TRAIN.VAL.rec_pretrained)
        aster = aster.to(self.device)
        aster = torch.nn.DataParallel(aster, device_ids=range(cfg.ngpu))
        return aster, aster_info

    def parse_aster_data(self, imgs_input):
        cfg = self.config.TRAIN
        aster_info = AsterInfo(cfg.voc_type)
        input_dict = {}
        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [aster_info.max_len] * batch_size
        return input_dict


class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)
