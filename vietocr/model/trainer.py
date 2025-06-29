import torch
import os
from tqdm.auto import tqdm
import torch.distributed as dist
from collections import OrderedDict

from vietocr.tool.logger import get_logger
from vietocr.tool.utils import download_file

class Trainer():
    def __init__(self, config, model, optimizer, criterion, local_rank=-1):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.local_rank = local_rank
        self.is_distributed = local_rank != -1
        self.is_main_process = local_rank in [-1, 0]
        self.device = config['device']

        self.epoch = 1
        self.step = 0
        self.best_acc = 0
        self.metrics = {}
        
        # Thiết lập logger chỉ cho process chính
        if self.is_main_process:
            self.checkpoint_dir = os.path.join(config['trainer']['checkpoint_dir'], config['experiment_name'])
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger = get_logger(os.path.join(self.checkpoint_dir, 'train.log'))
            self.logger.info('Created checkpoint directory: %s' % self.checkpoint_dir)
            self.logger.info('Loaded config: \n%s' % open(config['config_path']).read())

    def train(self, train_loader, valid_loader):
        if self.is_main_process:
            self.logger.info('Start training...')

        for self.epoch in range(self.epoch, self.config['trainer']['epochs'] + 1):
            
            # Thiết lập epoch cho sampler để đảm bảo shuffle khác nhau mỗi epoch
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.epoch)

            self.train_one_epoch(train_loader)

            # Chỉ process chính thực hiện validation và lưu model
            if self.is_main_process:
                self.logger.info('Epoch: %d' % self.epoch)
                self.validate(valid_loader)

                if self.metrics['full_seq_acc'] > self.best_acc:
                    self.best_acc = self.metrics['full_seq_acc']
                    self.save(os.path.join(self.checkpoint_dir, 'best_acc.pth'))
                
                self.save(os.path.join(self.checkpoint_dir, 'last.pth'))
                self.logger.info('Best acc: %.4f' % self.best_acc)

            # Chờ tất cả các process đồng bộ trước khi bắt đầu epoch mới
            if self.is_distributed:
                dist.barrier()

    def train_one_epoch(self, train_loader):
        self.model.train()
        
        pbar = tqdm(total=len(train_loader), desc="Train", disable=not self.is_main_process)
        
        for i, batch in enumerate(train_loader):
            self.step += 1
            
            img = batch['image'].to(self.device, non_blocking=True)
            tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
            tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
            tgt_padding_mask = batch['tgt_padding_mask'].to(self.device, non_blocking=True)

            outputs = self.model(img, tgt_input, tgt_padding_mask)
            
            outputs = outputs.view(-1, outputs.size(2))
            tgt_output = tgt_output.view(-1)
            
            loss = self.criterion(outputs, tgt_output)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['trainer']['clip_grad'])
            self.optimizer.step()
            
            # Chỉ process chính cập nhật progress bar và log
            if self.is_main_process:
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})
        
        pbar.close()

    def validate(self, data_loader):
        self.model.eval()
        pbar = tqdm(total=len(data_loader), desc="Validation", disable=not self.is_main_process)
        
        total_loss = 0
        total_correct_chars = 0
        total_chars = 0
        total_correct_seqs = 0
        total_seqs = 0
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                img = batch['image'].to(self.device, non_blocking=True)
                tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
                tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
                tgt_padding_mask = batch['tgt_padding_mask'].to(self.device, non_blocking=True)
                tgt_text = batch['tgt_text']

                outputs = self.model(img, tgt_input, tgt_padding_mask)
                
                loss = self.criterion(
                    outputs.view(-1, outputs.size(2)),
                    tgt_output.view(-1)
                )

                # model gốc khi dùng DDP
                model_to_eval = self.model.module if self.is_distributed else self.model
                
                if self.config['beamsearch']:
                    s = model_to_eval.beam_search(img, self.config)
                else:
                    s = model_to_eval.greedy_search(img, self.config)
                
                batch_correct_chars, batch_total_chars, batch_correct_seqs = self.calculate_acc(s, tgt_text)

                total_loss += loss.item() * len(tgt_text)
                total_correct_chars += batch_correct_chars
                total_chars += batch_total_chars
                total_correct_seqs += batch_correct_seqs
                total_seqs += len(tgt_text)

                pbar.update(1)
        
        pbar.close()
        
        # Đồng bộ hóa metrics từ tất cả các GPU
        if self.is_distributed:
            metrics_tensor = torch.tensor([total_loss, total_correct_chars, total_chars, total_correct_seqs, total_seqs]).to(self.device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            total_loss, total_correct_chars, total_chars, total_correct_seqs, total_seqs = metrics_tensor.tolist()

        # Chỉ process chính tính toán và log kết quả cuối cùng
        if self.is_main_process:
            val_loss = total_loss / total_seqs
            char_acc = total_correct_chars / total_chars
            full_seq_acc = total_correct_seqs / total_seqs
            
            self.logger.info('Validation loss: %.4f - char_acc: %.4f - full_seq_acc: %.4f' % (val_loss, char_acc, full_seq_acc))
            self.metrics = {'val_loss': val_loss, 'char_acc': char_acc, 'full_seq_acc': full_seq_acc}

    def calculate_acc(self, pred, target):
        correct_chars = 0
        total_chars = 0
        correct_seqs = 0
        
        for i in range(len(target)):
            if pred[i] == target[i]:
                correct_seqs += 1
            
            for j in range(min(len(pred[i]), len(target[i]))):
                if pred[i][j] == target[i][j]:
                    correct_chars += 1
            total_chars += len(target[i])
            
        return correct_chars, total_chars, correct_seqs

    def save(self, checkpoint_path):
        if not self.is_main_process:
            return

        # Khi dùng DDP, model được bọc trong module
        state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        
        state = {
            'model': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'epoch': self.epoch,
            'step': self.step,
            'best_acc': self.best_acc,
        }
        torch.save(state, checkpoint_path)
        self.logger.info('Saving checkpoint: %s' % checkpoint_path)

    def load(self, checkpoint_path):
        if 'http' in checkpoint_path:
            checkpoint_path = download_file(checkpoint_path, self.checkpoint_dir)
        
        # Ánh xạ checkpoint đến đúng GPU của process hiện tại
        map_location = f'cuda:{self.local_rank}' if self.is_distributed else self.device

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        saved_state_dict = checkpoint['model']
        new_state_dict = OrderedDict()
        
        # Xử lý prefix 'module.' khi load checkpoint DDP vào model thường hoặc ngược lại
        is_ddp_model = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
        saved_is_ddp = all([k.startswith('module.') for k in saved_state_dict.keys()])

        if is_ddp_model and not saved_is_ddp:
            # model hiện tại là DDP, checkpoint không phải -> thêm prefix
            for k, v in saved_state_dict.items():
                name = 'module.' + k
                new_state_dict[name] = v
        elif not is_ddp_model and saved_is_ddp:
            # model hiện tại không phải DDP, checkpoint là DDP -> bỏ prefix
            for k, v in saved_state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
        else: # Cả hai cùng là DDP hoặc cùng không phải
            new_state_dict = saved_state_dict

        self.model.load_state_dict(new_state_dict)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint.get('epoch', 1)
        self.step = checkpoint.get('step', 0)
        self.best_acc = checkpoint.get('best_acc', 0)
        
        if self.is_main_process:
            self.logger.info('Loading checkpoint from %s' % checkpoint_path)
