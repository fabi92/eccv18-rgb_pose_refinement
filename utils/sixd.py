

import os
import sys
import itertools

import numpy as np
import cv2
import yaml

from rendering.model import Model3D

''' The following were copied over from the 6DB toolkit'''
def load_yaml(path):
    with open(path, 'r') as f:
        content = yaml.load(f)
        return content


def load_info(path):
    with open(path, 'r') as f:
        info = yaml.load(f)
        for eid in info.keys():
            if 'cam_K' in info[eid].keys():
                info[eid]['cam_K'] = np.array(info[eid]['cam_K']).reshape((3, 3))
            if 'cam_R_w2c' in info[eid].keys():
                info[eid]['cam_R_w2c'] = np.array(info[eid]['cam_R_w2c']).reshape((3, 3))
            if 'cam_t_w2c' in info[eid].keys():
                info[eid]['cam_t_w2c'] = np.array(info[eid]['cam_t_w2c']).reshape((3, 1))
    return info

def load_gt(path):
    with open(path, 'r') as f:
        gts = yaml.load(f)
        for im_id, gts_im in gts.items():
            for gt in gts_im:
                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
    return gts

class Frame:
    def __init__(self):
        self.nr = None
        self.color = None
        self.depth = None
        self.cam = np.identity(3)
        self.gt = []


class Benchmark:
    def __init__(self):
        self.frames = []
        self.cam = np.identity(3)
        self.models = {}


def load_sixd(base_path, seq, nr_frames=0, load_mesh=True, subset_models=[]):

    bench = Benchmark()
    bench.scale_to_meters = 0.001
    if os.path.exists(os.path.join(base_path, 'camera.yml')):
        cam_info = load_yaml(os.path.join(base_path, 'camera.yml'))
        bench.cam[0, 0] = cam_info['fx']
        bench.cam[0, 2] = cam_info['cx']
        bench.cam[1, 1] = cam_info['fy']
        bench.cam[1, 2] = cam_info['cy']
        bench.scale_to_meters = 0.001 * cam_info['depth_scale']
    else:
        raise FileNotFoundError

    models_path = 'models'
    if not os.path.exists(os.path.join(base_path, models_path)):
        models_path = 'models_reconst'

    model_info = load_yaml(os.path.join(base_path, models_path, 'models_info.yml'))
    for key, val in model_info.items():
        bench.models[str(key)] = Model3D()
        bench.models[str(key)].diameter = val['diameter']

    if seq is None:
        return bench

    path = base_path + '/test/{:02d}/'.format(int(seq))
    info = load_info(path + 'info.yml')
    gts = load_gt(path + 'gt.yml')

    # Load frames
    nr_frames = nr_frames if nr_frames > 0 else len(info)
    for i in range(nr_frames):
        fr = Frame()
        fr.nr = i
        nr_string = '{:05d}'.format(i) if 'tudlight' in base_path else '{:04d}'.format(i)
        fr.color = cv2.imread(os.path.join(path, "rgb", nr_string + ".png")).astype(np.float32) / 255.0
        fr.depth = cv2.imread(os.path.join(path, "depth", nr_string + ".png"), -1).astype(np.float32)\
                   * bench.scale_to_meters
        if 'tless' in base_path:  # TLESS depth is in micrometers... why not nano? :D
            fr.depth *= 10
        if os.path.exists(os.path.join(path, 'mask')):
            fr.mask = cv2.imread(os.path.join(path, 'mask', nr_string + ".png"), -1)

        for gt in gts[i]:
            if subset_models and str(gt['obj_id']) not in subset_models:
                continue

            pose = np.identity(4)
            pose[:3, :3] = gt['cam_R_m2c']
            pose[:3, 3] = np.squeeze(gt['cam_t_m2c']) * bench.scale_to_meters
            if str(int(gt['obj_id'])) == str(int(seq)):
                fr.gt.append((str(gt['obj_id']), pose, gt['obj_bb']))

        fr.cam = info[i]['cam_K']
        bench.frames.append(fr)

    if load_mesh:
        # Build a set of all used model IDs for this sequence
        all_gts = list(itertools.chain(*[f.gt for f in bench.frames]))
        for ID in set([gt[0] for gt in all_gts]):
            bench.models[str(ID)].load(os.path.join(base_path, "models/obj_{:02d}.ply".format(int(ID))),
                                       scale_to_meter=bench.scale_to_meters)

    return bench


