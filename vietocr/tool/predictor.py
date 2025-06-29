from vietocr.tool.translate import (
    build_model,
    translate,
    translate_beam_search,
    process_input,
    predict,
)
from vietocr.tool.utils import download_weights
from vietocr.loader.prediction_loader import get_prediction_dataloader

import torch
import os
import json
from collections import defaultdict

class Predictor:
    def __init__(self, config):

        device = config["device"]

        model, vocab = build_model(config)
        weights = "/tmp/weights.pth"

        if config["weights"].startswith("http"):
            weights = download_weights(config["weights"])
        else:
            weights = config["weights"]

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab
        self.device = device

    def predict(self, img, return_prob=False):
        img = process_input(
            img,
            self.config["dataset"]["image_height"],
            self.config["dataset"]["image_min_width"],
            self.config["dataset"]["image_max_width"],
        )
        img = img.to(self.config["device"])

        if self.config["predictor"]["beamsearch"]:
            sent = translate_beam_search(img, self.model)
            s = sent
            prob = None
        else:
            s, prob = translate(img, self.model)
            s = s[0].tolist()
            prob = prob[0]

        s = self.vocab.decode(s)

        if return_prob:
            return s, prob
        else:
            return s

    def predict_batch(self, imgs, return_prob=False):
        bucket = defaultdict(list)
        bucket_idx = defaultdict(list)
        bucket_pred = {}

        sents, probs = [0] * len(imgs), [0] * len(imgs)

        for i, img in enumerate(imgs):
            img = process_input(
                img,
                self.config["dataset"]["image_height"],
                self.config["dataset"]["image_min_width"],
                self.config["dataset"]["image_max_width"],
            )

            bucket[img.shape[-1]].append(img)
            bucket_idx[img.shape[-1]].append(i)

        for k, batch in bucket.items():
            batch = torch.cat(batch, 0).to(self.device)
            s, prob = translate(batch, self.model)
            prob = prob.tolist()

            s = s.tolist()
            s = self.vocab.batch_decode(s)

            bucket_pred[k] = (s, prob)

        for k in bucket_pred:
            idx = bucket_idx[k]
            sent, prob = bucket_pred[k]
            for i, j in enumerate(idx):
                sents[j] = sent[i]
                probs[j] = prob[i]

        if return_prob:
            return sents, probs
        else:
            return sents

    def predict_dataloader(self, dataloader, return_prob=False):
        """
        Dự đoán trên một DataLoader.
        """
        all_sents = []
        all_probs = []
        all_paths = []

        for batched_data in dataloader:
            for batch in batched_data:
                imgs = batch['imgs'].to(self.device)
                paths = batch['paths']

                s, prob = translate(imgs, self.model)
                prob = prob.tolist()

                s = s.tolist()
                s = self.vocab.batch_decode(s)

                all_sents.extend(s)
                all_probs.extend(prob)
                all_paths.extend(paths)

        if return_prob:
            return all_sents, all_probs, all_paths
        else:
            return all_sents, all_paths
        
    def predict_folder(self, 
                   folder_path: str, 
                   batch_size: int = 32, 
                   return_prob: bool = True, 
                   output_path: str = None):
        """
        Dự đoán tất cả ảnh trong một thư mục.
        """
        # --- SỬA LỖI TẠI ĐÂY ---
        # 1. Tạo một đối tượng Vocab độc lập từ config.
        # Thao tác này an toàn vì self.config luôn tồn tại.

        image_paths = self._get_image_paths(folder_path)
        if not image_paths:
            print(f"Lỗi: Không tìm thấy file ảnh nào trong thư mục '{folder_path}'.")
            return []

        print(f"Tìm thấy {len(image_paths)} ảnh. Bắt đầu dự đoán với batch size = {batch_size}...")

        dataloader = get_prediction_dataloader(image_paths, self.config, batch_size)

        # 2. Truyền đối tượng vocab vừa tạo vào hàm _predict_dataloader
        unordered_results = self._predict_dataloader(dataloader, self.vocab, return_prob)

        final_results = []
        for path in image_paths:
            if path in unordered_results:
                result = unordered_results[path]
                final_results.append({
                    'path': path,
                    'filename': os.path.basename(path),
                    'prediction': result['prediction'],
                    'confidence': result['confidence']
                })
        
        print("Dự đoán hoàn tất!")

        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, ensure_ascii=False, indent=4)
                print(f"Kết quả đã được lưu thành công vào file: {output_path}")
            except Exception as e:
                print(f"Lỗi khi lưu file JSON: {e}")

        return final_results

    def _get_image_paths(self, folder_path: str):
        """Hàm nội bộ để lấy đường dẫn ảnh."""
        image_paths = []
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(supported_formats):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _predict_dataloader(self, dataloader, vocab, return_prob=True):
        """
        Hàm nội bộ để dự đoán trên dataloader.
        """
        results = {}
        for batched_data in dataloader:
            if batched_data is None:
                continue

            for batch in batched_data:
                imgs = batch['imgs'].to(self.device)
                paths = batch['paths']

                with torch.no_grad():
                    s, prob = translate(imgs, self.model)
                
                prob = prob.tolist()
                
                # --- SỬA LỖI TẠI ĐÂY ---
                # 3. Sử dụng đối tượng vocab đã được truyền vào để giải mã.
                s = self.vocab.batch_decode(s.tolist())

                for i in range(len(paths)):
                    prediction = s[i]
                    confidence = prob[i] if return_prob else None
                    results[paths[i]] = {'prediction': prediction, 'confidence': confidence}
        return results