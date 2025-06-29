import torch
import os
import argparse
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
from vietocr.loader.dataloader import OCRDataset
from vietocr.model.vocab import Vocab
from vietocr.model.transformerocr import VietOCR
from vietocr.optim.optim import ScheduledOptim
from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='see config file for details')
    parser.add_argument('--checkpoint', required=False, help='your checkpoint')
    parser.add_argument('--no-aug', action='store_true', help='disable augmentation')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)
    
    # Kiểm tra xem có đang chạy trong môi trường distributed hay không
    # torchrun sẽ tự động thiết lập các biến môi trường này.
    is_distributed = 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1

    local_rank = -1
    if is_distributed:
        # Lấy các thông tin cần thiết từ biến môi trường
        world_size = int(os.environ['WORLD_SIZE'])
        global_rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Khởi tạo process group
        dist.init_process_group(backend='nccl')
        
        # Thiết lập device cho process hiện tại
        torch.cuda.set_device(local_rank)
        print(f"[{os.getpid()}] Initializing DDP on rank {global_rank}, local rank {local_rank}, world size {world_size}.")
    
    # Thiết lập seed để đảm bảo tính nhất quán
    seed = config['trainer'].get('seed', 1337)
    seed_everything(seed)

    # Thiết lập device
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        # Giữ nguyên logic cũ nếu không phải distributed
        device = torch.device(config['device'])
    
    config['device'] = device

    # Khởi tạo vocab
    vocab = Vocab(config['vocab'])
    config['vocab_size'] = len(vocab)
    
    # Khởi tạo model
    model = VietOCR(config)
    model.to(device)

    # Bọc model với DistributedDataParallel nếu cần
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Tạo dataset
    train_aug = config['aug']['train'] if not args.no_aug else config['aug']['valid']
    
    train_dataset = OCRDataset(img_path=config['dataset']['train_img_root'],
                               annotation_path=config['dataset']['train_annotation'],
                               vocab=vocab, transform=train_aug,
                               img_height=config['image']['height'],
                               img_width=config['image']['width'],
                               is_padding=config['image'].get('padding', False))

    valid_dataset = OCRDataset(img_path=config['dataset']['valid_img_root'],
                               annotation_path=config['dataset']['valid_annotation'],
                               vocab=vocab, transform=config['aug']['valid'],
                               img_height=config['image']['height'],
                               img_width=config['image']['width'],
                               is_padding=config['image'].get('padding', False))
    
    # Tạo DistributedSampler nếu cần
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        # Sử dụng sampler cho validation để chia nhỏ dữ liệu trên các GPU
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
    else:
        train_sampler = None
        valid_sampler = None

    # Tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['trainer']['batch_size'],
        shuffle=(train_sampler is None), # shuffle=False khi dùng sampler
        num_workers=config['trainer']['num_workers'],
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=train_dataset.collate_fn)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['trainer']['batch_size'],
        shuffle=(valid_sampler is None),
        num_workers=config['trainer']['num_workers'],
        pin_memory=True,
        sampler=valid_sampler,
        collate_fn=valid_dataset.collate_fn)

    # Khởi tạo loss và optimizer
    criterion = LabelSmoothingLoss(len(vocab), padding_idx=vocab.pad, smoothing=0.1).to(device)
    
    # Lấy model gốc khi dùng DDP
    model_to_optimize = model.module if is_distributed else model
    optimizer = ScheduledOptim(
        model_to_optimize.parameters(), 
        config['transformer']['d_model'], 
        config['optimizer']['n_warmup_steps'])

    # Khởi tạo Trainer
    trainer = Trainer(config, model, optimizer, criterion, local_rank=local_rank)

    if args.checkpoint:
        trainer.load(args.checkpoint)
    
    trainer.train(train_loader, valid_loader)

    # Dọn dẹp process group
    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
