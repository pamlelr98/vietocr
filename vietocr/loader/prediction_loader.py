import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import defaultdict
import numpy as np
from vietocr.tool.translate import process_image

class PredictionDataset(Dataset):
    """
    Dataset để load ảnh cho việc predict.
    """
    def __init__(self, image_paths, image_height, image_min_width, image_max_width):
        self.image_paths = image_paths
        self.image_height = image_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Lỗi khi mở file ảnh: {image_path}. Lỗi: {e}")
            return None 

        processed_img = process_image(img, self.image_height, self.image_min_width, self.image_max_width)
        # Chuyển đổi sang float32 để khớp với kiểu dữ liệu của mô hình
        return {'img': processed_img.astype(np.float32), 'path': image_path}

def prediction_collate_fn(batch):
    """
    Hàm collate để nhóm các ảnh có cùng chiều rộng lại với nhau (bucketing).
    """
    # Lọc ra các sample bị lỗi (None)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    bucket = defaultdict(list)
    bucket_paths = defaultdict(list)

    for sample in batch:
        img = sample['img']
        path = sample['path']
        bucket[img.shape[-1]].append(torch.from_numpy(img))
        bucket_paths[img.shape[-1]].append(path)

    batched_data = []
    for width in bucket:
        # Sử dụng torch.stack để tạo batch dimension, kết quả là (N, C, H, W)
        imgs = torch.stack(bucket[width], 0)
        paths = bucket_paths[width]
        batched_data.append({'imgs': imgs, 'paths': paths})

    return batched_data

def get_prediction_dataloader(image_paths, config, batch_size=1):
    """
    Hàm tạo DataLoader cho việc predict.
    """
    dataset = PredictionDataset(
        image_paths,
        config['dataset']['image_height'],
        config['dataset']['image_min_width'],
        config['dataset']['image_max_width']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # Không xáo trộn để giữ thứ tự
        num_workers=config['dataloader'].get('num_workers', 0), # Dùng .get để an toàn
        collate_fn=prediction_collate_fn,
        pin_memory=config['dataloader'].get('pin_memory', False)
    )
    return dataloader