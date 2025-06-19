import os
import sys
import glob
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from mmcv.image import imread

from mmocr.apis.inferencer import MMOCRInferencer

def build_inferencer(config_file, checkpoint_file, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return MMOCRInferencer(det=None, recog=None, kie=None, config=config_file, ckpt=checkpoint_file, device=device)

class Inference:
    def __init__(self, config_file, checkpoint_file, device=None):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.device = device
        self.inferencer = build_inferencer(config_file, checkpoint_file, device=device)

    def result_format(self, pred, file_path):
        raise NotImplementedError

    def predict_single_file(self, file_path):
        pass

    def predict_batch(self, imgs):
        pass

class Structure_Recognition(Inference):
    def __init__(self, config_file, checkpoint_file, samples_per_gpu=4, device=None):
        super().__init__(config_file, checkpoint_file, device=device)
        self.samples_per_gpu = samples_per_gpu

    def result_format(self, pred, file_path=None):
        # pred is a dict from MMOCRInferencer
        return pred

    def predict_single_file(self, file_path):
        img = imread(file_path)
        file_name = os.path.basename(file_path)
        result = self.inferencer(img, batch_size=1)
        result = self.result_format(result, file_path)
        result_dict = {file_name: result}
        return result, result_dict

class Runner:
    def __init__(self, cfg):
        self.structure_master_config = cfg['structure_master_config']
        self.structure_master_ckpt = cfg['structure_master_ckpt']
        self.structure_master_result_folder = cfg['structure_master_result_folder']

        test_folder = cfg['test_folder']
        chunks_nums = cfg['chunks_nums']
        self.chunks_nums = chunks_nums
        self.chunks = self.get_file_chunks(test_folder, chunks_nums=chunks_nums)

    def init_structure_master(self):
        self.master_structure_inference = Structure_Recognition(
            self.structure_master_config, self.structure_master_ckpt)

    def release_structure_master(self):
        torch.cuda.empty_cache()
        del self.master_structure_inference

    def do_structure_predict(self, path, is_save=True, gpu_idx=None):
        if isinstance(path, str):
            if os.path.isfile(path):
                all_results = dict()
                print('Single file in structure master prediction ...')
                _, result_dict = self.master_structure_inference.predict_single_file(path)
                all_results.update(result_dict)
            elif os.path.isdir(path):
                all_results = dict()
                print('Folder files in structure master prediction ...')
                search_path = os.path.join(path, '*.png')
                files = glob.glob(search_path)
                for file in tqdm(files):
                    _, result_dict = self.master_structure_inference.predict_single_file(file)
                    all_results.update(result_dict)
            else:
                raise ValueError
        elif isinstance(path, list):
            all_results = dict()
            print('Chunks files in structure master prediction ...')
            for i, p in enumerate(path):
                _, result_dict = self.master_structure_inference.predict_single_file(p)
                all_results.update(result_dict)
                if gpu_idx is not None:
                    print(f"[GPU_{gpu_idx} : {i+1} / {len(path)}] {p} file structure inference.")
                else:
                    print(f"{p} file structure inference.")
        else:
            raise ValueError

        if is_save:
            if not os.path.exists(self.structure_master_result_folder):
                os.makedirs(self.structure_master_result_folder)
            if not isinstance(path, list):
                save_file = os.path.join(self.structure_master_result_folder, 'structure_master_results.pkl')
            else:
                save_file = os.path.join(self.structure_master_result_folder, f'structure_master_results_{gpu_idx}.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(all_results, f)

    def get_file_chunks(self, folder, chunks_nums=8):
        print("Divide files to chunks for multiply gpu device inference.")
        file_paths = glob.glob(os.path.join(folder, '*.png'))
        counts = len(file_paths)
        nums_per_chunk = counts // chunks_nums
        img_chunks = []
        for n in range(chunks_nums):
            if n == chunks_nums - 1:
                s = n * nums_per_chunk
                img_chunks.append(file_paths[s:])
            else:
                s = n * nums_per_chunk
                e = (n + 1) * nums_per_chunk
                img_chunks.append(file_paths[s:e])
        return img_chunks

    def run_structure_single_chunk(self, chunk_id):
        paths = self.chunks[chunk_id]
        self.init_structure_master()
        self.do_structure_predict(paths, is_save=True, gpu_idx=chunk_id)
        self.release_structure_master()

if __name__ == '__main__':
    chunk_nums = int(sys.argv[1])
    chunk_id = int(sys.argv[2])
    epoch_id = int(sys.argv[3])
    val_test = sys.argv[4]

    cfg = {
        'structure_master_config': './configs/textrecog/master/table_master_ResnetExtract_Ranger_0705_FinTabNet_cell150_batch4.py',
        'structure_master_ckpt': f'/home2/nam/nam_data/work_dir/1114_TableMASTER_FinTabNet_seq500_cell150_batch4/epoch_{epoch_id}.pth',
        'structure_master_result_folder': f'/home2/nam/nam_data/work_dir/1114_TableMASTER_FinTabNet_seq500_cell150_batch4/structure_{val_test}_result_epoch_{epoch_id}',
        'test_folder': f'/disks/strg16-176/nam/data/fintabnet/img_tables/{val_test}/',
        'chunks_nums': chunk_nums
    }

    print(cfg)
    if not os.path.exists(cfg['structure_master_result_folder']):
        os.makedirs(cfg['structure_master_result_folder'], exist_ok=True)
    with open(cfg['structure_master_result_folder'] + '/cfg.txt', 'w') as f:
        f.write(cfg['structure_master_config'] + '\n')
        f.write(cfg['structure_master_ckpt'] + '\n')
        f.write(cfg['structure_master_result_folder'] + '\n')
        f.write(cfg['test_folder'] + '\n')

    runner = Runner(cfg)
    runner.run_structure_single_chunk(chunk_id=chunk_id)