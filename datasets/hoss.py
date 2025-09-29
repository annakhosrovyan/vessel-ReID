# encoding: utf-8
import glob
import os.path as osp
from .bases import BaseImageDataset


class HOSS(BaseImageDataset):
    dataset_dir = 'HOSS'
    s2o_dir = 'subset/S2O'
    o2s_dir = 'subset/O2S'
    
    def __init__(self, root='', verbose=True, pid_begin = 0, eval_mode='all', **kwargs):
        super(HOSS, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.s2o_dir = osp.join(self.dataset_dir, self.s2o_dir)
        self.o2s_dir = osp.join(self.dataset_dir, self.o2s_dir)

        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        
        self.query_val_dir = osp.join(self.dataset_dir, 'query_val')
        self.gallery_val_dir = osp.join(self.dataset_dir, 'gallery_val')

        if eval_mode == 's2o':
            self.query_dir = osp.join(self.s2o_dir, 'query')
            self.gallery_dir = osp.join(self.s2o_dir, 'bounding_box_test')
        elif eval_mode == 'o2s':
            self.query_dir = osp.join(self.o2s_dir, 'query')
            self.gallery_dir = osp.join(self.o2s_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train, train_pair = self._process_dir_train(self.train_dir, relabel=True)
        
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        
        query_val = self._process_dir(self.query_val_dir, relabel=False)
        gallery_val = self._process_dir(self.gallery_val_dir, relabel=False)

        if eval_mode == 's2s':
            query = self._filter_by_modality(query, modality='SAR')
            gallery = self._filter_by_modality(gallery, modality='SAR')
            query, gallery = self._ensure_mutual_ids(query, gallery)
        elif eval_mode == 'o2o':
            query = self._filter_by_modality(query, modality='RGB')
            gallery = self._filter_by_modality(gallery, modality='RGB')
            query, gallery = self._ensure_mutual_ids(query, gallery)

        if verbose:
            print("=> HOSS ReID Dataset loaded")
            self.print_dataset_statistics(train, query, gallery, query_val, gallery_val)
            
            if train_pair is not None:
                print("Number of RGB-SAR pair: {}".format(len(train_pair)))
                print("  ----------------------------------------")

        self.train = train
        self.train_pair = train_pair
        self.query = query
        self.gallery = gallery
        self.query_val = query_val
        self.gallery_val = gallery_val

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_train_pair_pids, self.num_train_pair_imgs, self.num_train_pair_cams, self.num_train_pair_vids = self.get_imagedata_info_pair(self.train_pair)

        self.num_query_pids_val, self.num_query_imgs_val, self.num_query_cams_val, self.num_query_vids_val = self.get_imagedata_info(self.query_val)
        self.num_gallery_pids_val, self.num_gallery_imgs_val, self.num_gallery_cams_val, self.num_gallery_vids_val = self.get_imagedata_info(self.gallery_val)

        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _ensure_mutual_ids(self, query, gallery):
        query_ids = set()
        gallery_ids = set()
        
        for _, pid, _, _ in query:
            query_ids.add(pid)
        for _, pid, _, _ in gallery:
            gallery_ids.add(pid)
        
        mutual_ids = query_ids.intersection(gallery_ids)
        
        filtered_query = [(img_path, pid, camid, trackid) for img_path, pid, camid, trackid in query if pid in mutual_ids]
        filtered_gallery = [(img_path, pid, camid, trackid) for img_path, pid, camid, trackid in gallery if pid in mutual_ids]
        
        return filtered_query, filtered_gallery

    def _filter_by_modality(self, dataset, modality):
        filtered_dataset = []
        for img_path, pid, camid, trackid in dataset:
            if modality == 'SAR' and img_path.endswith('SAR.tif'):
                filtered_dataset.append((img_path, pid, camid, trackid))
            elif modality == 'RGB' and img_path.endswith('RGB.tif'):
                filtered_dataset.append((img_path, pid, camid, trackid))
        return filtered_dataset

    def get_imagedata_info_pair(self, data):
        pids, cams, tracks = [], [], []

        for img in data:
            for _, pid, camid, trackid in img:
                pids += [pid]
                cams += [camid]
                tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.query_val_dir):
            raise RuntimeError("'{}' is not available".format(self.query_val_dir))
        if not osp.exists(self.gallery_val_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_val_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.tif'))

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[0])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[0])
            # camid 0 for RGB, 1 for SAR
            camid = 0 if img_path.split('/')[-1].split('_')[-1] == 'RGB.tif' else 1
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        return dataset

    def _process_dir_train(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.tif'))

        RGB_paths = [i for i in img_paths if i.endswith('RGB.tif')]
        pid2sar = {}

        pid_container = set()
        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[0])
            pid_container.add(pid)
            if img_path.endswith('SAR.tif'):
                if pid not in pid2sar:
                    pid2sar[pid] = [img_path]
                else:
                    pid2sar[pid].append(img_path)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(img_paths):
            pid = int(img_path.split('/')[-1].split('_')[0])
            # camid 0 for RGB, 1 for SAR
            camid = 0 if img_path.split('/')[-1].split('_')[-1] == 'RGB.tif' else 1
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        dataset_pair = []
        for img_path in sorted(RGB_paths):
            pid = int(img_path.split('/')[-1].split('_')[0])
            if pid not in pid2sar.keys():
                continue
            for sar_path in pid2sar[pid]:
                dataset_pair.append([(img_path, self.pid_begin + pid, 0, 1),
                                     (sar_path, self.pid_begin + pid, 1, 1)])

        return dataset, dataset_pair