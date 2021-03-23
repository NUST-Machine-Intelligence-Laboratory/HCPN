
import torch
from torch.utils import data
from torchvision import transforms

import os
import time
import random
import numpy as np

from args import get_parser
from libs.dataset.data import get_dataset

from libs.utils.obj import Loss
from libs.utils.eval import db_eval_iou_multi
from libs.utils.utils import make_dir
from libs.utils.utils import get_optimizer
from libs.utils.utils import check_parallel, save_checkpoint_epoch, load_checkpoint_epoch
from libs.model.HCPN import EncoderNet, DecoderNet


def init_dataloaders(args):
    loaders = {}

    # init dataloaders for training and validation
    for split in ['train', 'val']:
        batch_size = args.batch_size
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image_transforms = transforms.Compose([to_tensor, normalize])
        target_transforms = transforms.Compose([to_tensor])

        dataset = get_dataset(
            args, split=split, image_transforms=image_transforms,
            target_transforms=target_transforms,
            augment=args.augment and split == 'train',
            input_size=(512, 512), pre_train=pretrain)

        shuffle = True if split == 'train' else False
        loaders[split] = data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=args.num_workers,
                                         drop_last=True,
                                         pin_memory=True,
                                         )

    return loaders


def trainIters(args):
    print(args)

    model_dir = os.path.join('model/', args.model_name)
    make_dir(model_dir)

    epoch_resume = 0
    if args.resume:
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = \
            load_checkpoint_epoch(args.model_name, args.epoch_resume,
                                  args.use_gpu)

        epoch_resume = args.epoch_resume

        encoder = EncoderNet()
        decoder = DecoderNet()

        encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)
    else:
        encoder = EncoderNet()
        decoder = DecoderNet()

    criterion = Loss()

    if args.ngpus > 1 and args.use_gpu:
        decoder = torch.nn.DataParallel(decoder, device_ids=[0, 1], output_device=0)
        encoder = torch.nn.DataParallel(encoder, device_ids=[0, 1], output_device=0)
        criterion = torch.nn.DataParallel(criterion, device_ids=[0, 1], output_device=0)

    if args.use_gpu:
        encoder.cuda()
        decoder.cuda()
        criterion.cuda()

    if args.use_gpu:
        torch.cuda.synchronize()

    encoder_params = list(encoder.parameters())
    decoder_params = list(decoder.parameters())
    dec_opt = get_optimizer(args.optim, args.lr, decoder_params,
                            args.weight_decay)
    enc_opt = get_optimizer(args.optim_cnn, args.lr_cnn, encoder_params,
                            args.weight_decay_cnn)

    loaders = init_dataloaders(args)

    best_iou = 0

    start = time.time()

    for e in range(0, args.max_epoch):
        print("Epoch", e+1)
        epoch_losses = {'train': {'total': [], 'iou': [],
                                  'mask_loss': [], 'ctr_loss': []},
                        'val': {'total': [], 'iou': [],
                                'mask_loss': [], 'ctr_loss': []}}

        for split in ['train', 'val']:
            if split == 'train':
                encoder.train(True)
                decoder.train(True)
            else:
                encoder.train(False)
                decoder.train(False)

            for batch_idx, (im1, im2, flow, ms1, ms2, ctr1, ctr2, negative_pixels1, negative_pixels2) in\
                    enumerate(loaders[split]):
                # print(batch_idx)
                im1, im2, flow, mask1, mask2, ctr1, ctr2, negative_pixels1, negative_pixels2 = \
                    im1.cuda(), im2.cuda(), flow.cuda(), ms1.cuda(), ms2.cuda(), \
                    ctr1.cuda(), ctr2.cuda(), negative_pixels1.cuda(), negative_pixels2.cuda()

                if split == 'train':
                    h5_1, h4_1, h3_1, h2_1, h5_2, h4_2, h3_2, h2_2 = encoder(im1, im2, flow)
                    mask_1, c_1, mask_2, c_2 = decoder(h5_1, h4_1, h3_1, h2_1, h5_2, h4_2, h3_2, h2_2)

                    mask_loss1 = criterion(mask_1, mask1, negative_pixels1)
                    ctr_loss1 = criterion(c_1, ctr1, negative_pixels1)
                    mask_loss2 = criterion(mask_2, mask2, negative_pixels2)
                    ctr_loss2 = criterion(c_2, ctr2, negative_pixels2)

                    mask_loss = 0.5 * (mask_loss1 + mask_loss2)
                    ctr_loss = 0.5 * (ctr_loss1 + ctr_loss2)
                    loss = mask_loss + 0.5 * ctr_loss

                    iou = db_eval_iou_multi(mask1.cpu().detach().numpy(), mask_1.cpu().detach().numpy())

                    dec_opt.zero_grad()
                    enc_opt.zero_grad()
                    decoder.zero_grad()
                    loss.mean().backward()
                    enc_opt.step()
                    dec_opt.step()

                else:
                    with torch.no_grad():
                        h5_1, h4_1, h3_1, h2_1, h5_2, h4_2, h3_2, h2_2 = encoder(im1, im2, flow)
                        mask_1, c_1, mask_2, c_2 = decoder(h5_1, h4_1, h3_1, h2_1, h5_2, h4_2, h3_2, h2_2)

                        mask_loss1 = criterion(mask_1, mask1, negative_pixels1)
                        ctr_loss1 = criterion(c_1, ctr1, negative_pixels1)
                        mask_loss2 = criterion(mask_2, mask2, negative_pixels2)
                        ctr_loss2 = criterion(c_2, ctr2, negative_pixels2)

                        mask_loss = 0.5 * (mask_loss1 + mask_loss2)
                        ctr_loss = 0.5 * (ctr_loss1 + ctr_loss2)
                        loss = mask_loss + 0.5 * ctr_loss

                    iou = db_eval_iou_multi(mask1.cpu().detach().numpy(),
                                            mask_1.cpu().detach().numpy())

                epoch_losses[split]['total'].append(loss.data.mean().item())
                epoch_losses[split]['mask_loss'].append(mask_loss.data.mean().item())
                epoch_losses[split]['ctr_loss'].append(ctr_loss.data.mean().item())
                epoch_losses[split]['iou'].append(iou)

                if (batch_idx + 1) % args.print_every == 0:
                    mt = np.mean(epoch_losses[split]['total'])
                    mmask = np.mean(epoch_losses[split]['mask_loss'])
                    mbdry = np.mean(epoch_losses[split]['ctr_loss'])
                    miou = np.mean(epoch_losses[split]['iou'])

                    te = time.time() - start
                    print('Epoch: [{}/{}][{}/{}]\tTime {:.3f}s\tLoss: {:.4f}'
                          '\tMask Loss: {:.4f}\tCTR Loss: {:.4f}'
                          '\tIOU: {:.4f}'.format(e+1, args.max_epoch, batch_idx,
                                                 len(loaders[split]), te, mt,
                                                 mmask, mbdry, miou))

                    start = time.time()

        miou = np.mean(epoch_losses['val']['iou'])
        print(miou)

        if pretrain:
            save_checkpoint_epoch(args, encoder, decoder,
                                  enc_opt, dec_opt, miou, False)
        else:
            if miou > best_iou:
                best_iou = miou
                save_checkpoint_epoch(args, encoder, decoder,
                                      enc_opt, dec_opt, e+1, False)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    args.model_name = 'HCPN'
    args.batch_size = 10
    args.ngpus = 2
    args.max_epoch = 25
    args.year = '2016'

    args.num_works = 16

    # first pretrain on YouTube-VOS (pretrain = True).
    # max_epoch all sets to 25.
    pretrain = True

    # then train on davis dataset (pretrain = False)
    # epoch_resume set to miou and uncomment the following two lines.
    # args.resume = True
    # args.epoch_resume = 0.6513122544336745 # miou

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    torch.cuda.manual_seed(args.seed)

    trainIters(args)
