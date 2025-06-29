import os
import logging

class Logger:
    def __init__(self, fname):
        path, _ = os.path.split(fname)
        os.makedirs(path, exist_ok=True)

        self.logger = open(fname, "w")

    def log(self, string):
        self.logger.write(string + "\n")
        self.logger.flush()

    def close(self):
        self.logger.close()

def get_logger(log_file='train.log'):
    """
    Tạo và cấu hình một logger.
    
    Args:
        log_file (str): Đường dẫn đến file log.
        
    Returns:
        logging.Logger: Đối tượng logger đã được cấu hình.
    """
    # Lấy tên logger dựa trên tên file, hoặc một tên chung
    logger = logging.getLogger('VietOCR')
    logger.setLevel(logging.INFO)

    # Tránh thêm handler nhiều lần nếu logger đã được cấu hình
    if not logger.handlers:
        # Tạo handler cho console (stream)
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        
        # Tạo handler cho file
        f_handler = logging.FileHandler(log_file, mode='a')
        f_handler.setLevel(logging.INFO)

        # Định dạng cho log message
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)

        # Thêm handlers vào logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger