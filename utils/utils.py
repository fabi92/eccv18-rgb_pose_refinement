import yaml, cv2, glob
import numpy as np

from ..rendering.model import Model3D
from ..rendering.utils import perturb_pose


class Benchmark(object):
    def __init__(self):
        self.model = None
        self.frames = []


class Frame(object):
    def __init__(self, nr=-1, colorfile=None, depthfile=None, color=None, depth=None):
        self.nr = nr
        self.colorfile = colorfile
        self.detphfile = depthfile
        self.color = color
        self.depth = depth
        self.gt = []


def loadSIXDBench(dataset_path, seq, seq_to_name, nr_frames=-1, metric_crop_shape=None, rot_variation=0.174,
                  trans_variation=[0.2, 0.2, 0.2]):
    assert seq in seq_to_name.keys()  # Make sure that there is no typo

    model_path = dataset_path + "/models/obj_%02d" % seq_to_name[seq]

    bench = Benchmark()
    bench.model = Model3D()
    bench.model.load(model_path + ".ply", demean=False, scale_to_meter=0.001)
    bench.metric_cropshape = metric_crop_shape

    # Find min/max frame numbers
    color_path = dataset_path + "/test/%02d/rgb/" % seq_to_name[seq]
    depth_path = dataset_path + "/test/%02d/depth/" % seq_to_name[seq]

    color_files = glob.glob(color_path + '/*')

    # Load frames
    max_frames = len(color_files) - 1

    if nr_frames == -1:
        ran = range(max_frames - 1)
    else:
        ran = np.random.randint(0, max_frames, nr_frames)

    poses = yaml.load(open(dataset_path + "/test/%02d/gt.yml" % seq_to_name[seq]))

    for i in ran:
        fr = Frame()
        fr.nr = i
        fr.colorfile = color_path + "%04d.png" % i
        fr.depthfile = depth_path + "%04d.png" % i
        fr.color = cv2.imread(fr.colorfile).astype(np.float32) / 255.0
        fr.depth = cv2.imread(fr.depthfile, -1)
        fr.depth = 0.001 * fr.depth.astype(np.float32)

        pose = np.identity(4)
        samples = []

        for p in poses[i]:
            if p["obj_id"] == seq_to_name[seq]:
                pose[:3, :3] = np.array(p["cam_R_m2c"]).reshape(3, 3)
                pose[:3, 3] = np.array(p["cam_t_m2c"]) / 1000.
                break

        perturbed_pose = [perturb_pose(pose, rot_variation=rot_variation, trans_variation=trans_variation)]
        samples.append((pose, perturbed_pose))

        fr.gt.append((seq, samples))
        bench.frames.append(fr)

    return bench
