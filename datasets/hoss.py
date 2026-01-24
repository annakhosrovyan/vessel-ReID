# encoding: utf-8
import glob
import os.path as osp
from .bases import BaseImageDataset


class HOSS(BaseImageDataset):
    dataset_dir = 'HOSS_balanced'    
    def __init__(self, 
                root='/nfs/h100/raid/rs/vessel_detection', 
                verbose=True, 
                pid_begin=0, 
                eval_mode='all', 
                **kwargs
                ):
        super(HOSS, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        eval_modes = ['rgb_sar', 'sar_rgb', 'rgb_mixed', 'sar_mixed']
        if eval_mode == 'all':
            eval_mode_list = eval_modes
        else:
            eval_mode_list = [eval_mode]

        self.train_dir = osp.join(self.dataset_dir, 'train')
        
        self.query_dir = [osp.join(self.dataset_dir, 'test', mode, 'query') 
                            for mode in eval_mode_list]
        self.gallery_dir = osp.join(self.dataset_dir, 'test', 'gallery')
        
        self.query_val_dir = [osp.join(self.dataset_dir, 'val', mode, 'query') 
                              for mode in eval_mode_list]
        self.gallery_val_dir = osp.join(self.dataset_dir, 'val', 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin
        train, train_pair = self._process_dir_train(self.train_dir, relabel=True)
        
        query = []
        for query_dir in self.query_dir:
            query.extend(self._process_dir(query_dir, relabel=False))
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        
        query_val = []
        for query_val_dir in self.query_val_dir:
            query_val.extend(self._process_dir(query_val_dir, relabel=False))
        gallery_val = self._process_dir(self.gallery_val_dir, relabel=False)

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

    def print_dataset_statistics(self, train, query, gallery, query_val, gallery_val):
        num_train_pids, num_train_imgs, num_train_cams, _ = self.get_imagedata_info(train) if train is not None else (0, 0, 0, 0)

        eval_modes = ['rgb_sar', 'sar_rgb', 'rgb_mixed', 'sar_mixed']

        def summarize_split(query_dir, gallery_dir):
            query_data = self._process_dir(query_dir, relabel=False)
            gallery_data = self._process_dir(gallery_dir, relabel=False)
            num_query_pids, num_query_imgs, num_query_cams, _ = self.get_imagedata_info(query_data)
            num_gallery_pids, num_gallery_imgs, num_gallery_cams, _ = self.get_imagedata_info(gallery_data)
            return (num_query_pids, num_query_imgs, num_query_cams,
                    num_gallery_pids, num_gallery_imgs, num_gallery_cams)

        print("Dataset statistics:")
        print("  -------------------------------------------------------------")
        print("  subset     | eval_mode | # ids | # images | # cameras")
        print("  -------------------------------------------------------------")
        if train is not None:
            print("  train      |    all    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        for mode in eval_modes:
            test_query_dir = osp.join(self.dataset_dir, 'test', mode, 'query')
            val_query_dir = osp.join(self.dataset_dir, 'val', mode, 'query')
            test_gallery_dir = osp.join(self.dataset_dir, 'test', 'gallery')
            val_gallery_dir = osp.join(self.dataset_dir, 'val', 'gallery')
            (num_query_pids, num_query_imgs, num_query_cams,
             num_gallery_pids, num_gallery_imgs, num_gallery_cams) = summarize_split(test_query_dir, test_gallery_dir)
            (num_query_val_pids, num_query_val_imgs, num_query_val_cams,
             num_gallery_val_pids, num_gallery_val_imgs, num_gallery_val_cams) = summarize_split(val_query_dir, val_gallery_dir)

            print("  query      | {:8s} | {:5d} | {:8d} | {:9d}".format(mode, num_query_pids, num_query_imgs, num_query_cams))
            print("  gallery    | {:8s} | {:5d} | {:8d} | {:9d}".format(mode, num_gallery_pids, num_gallery_imgs, num_gallery_cams))
            print("  query_val  | {:8s} | {:5d} | {:8d} | {:9d}".format(mode, num_query_val_pids, num_query_val_imgs, num_query_val_cams))
            print("  gallery_val| {:8s} | {:5d} | {:8d} | {:9d}".format(mode, num_gallery_val_pids, num_gallery_val_imgs, num_gallery_val_cams))
            print("  -------------------------------------------------------------")

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
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        for query_dir in self.query_dir:
            if not osp.exists(query_dir):
                raise RuntimeError("'{}' is not available".format(query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        for query_val_dir in self.query_val_dir:
            if not osp.exists(query_val_dir):
                raise RuntimeError("'{}' is not available".format(query_val_dir))
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