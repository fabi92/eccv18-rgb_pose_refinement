"""
Simple script to run a forward pass employing the Refiner on a SIXD dataset sample with a trained model.

Usage:
  test_refinement.py [options]
  test_refinement.py -h | --help

Options:
    -d --dataset=<string>        Path to SIXD dataset
    -o --object=<string>         Object to be evaluated [default: 02]
    -n --network=<string>        Path to trained network [default: models/refiner_linemod_obj_02.pb]
    -r --max_rot_pert=<float>    Max. Rotational Perturbation to be applied in Degrees [default: 20.0]
    -t --max_trans_pert=<float>  Max. Translational Perturbation to be applied in Meters [default: 0.10]
    -i --iterations=<int>        Max. number of iterations
    -h --help                    Show this message and exit
"""

import tensorflow as tf
import yaml
import cv2
import numpy as np

from utils.sixd import load_sixd
from refiner.architecture import Architecture
from rendering.renderer import Renderer
from refiner.refiner import Refiner, Refinable
from rendering.utils import perturb_pose, trans_rot_err

from timeit import default_timer as timer
from docopt import docopt

args = docopt(__doc__)

sixd_base = args["--dataset"]
obj = args["--object"]
network = args["--network"]
max_rot_pert = float(args["--max_rot_pert"]) / 180. * np.pi
max_trans_pert = float(args["--max_trans_pert"])
iterations = int(args["--iterations"])

bench = load_sixd(sixd_base, nr_frames=1, seq=obj)
croppings = yaml.load(open('config/croppings.yaml', 'r'))

if 'linemod' in network:
    dataset_name = 'linemod'
elif 'tejani' in network:
    dataset_name = 'tejani'
else:
    raise Exception('Could not determine dataset')

with tf.Session() as session:
    architecture = Architecture(network_file=network, sess=session)
    ren = Renderer((640, 480), bench.cam)
    refiner = Refiner(architecture=architecture, ren=ren, session=session)

    for frame in bench.frames:
        col = frame.color.copy()
        _, gt_pose, _ = frame.gt[0]

        perturbed_pose = perturb_pose(gt_pose, max_rot_pert, max_trans_pert)

        refinable = Refinable(model=bench.models[str(int(obj))], label=0, hypo_pose=perturbed_pose,
                              metric_crop_shape=croppings[dataset_name]['obj_{:02d}'.format(int(obj))], input_col=col)

        for i in range(iterations):
            refinable.input_col = col.copy()

            start = timer()
            refiner.iterative_contour_alignment(refinable=refinable, max_iterations=1)
            end = timer()

            # Rendering of results
            ren.clear()
            ren.draw_background(col)
            ren.draw_boundingbox(refinable.model, refinable.hypo_pose)
            ren.draw_model(refinable.model, refinable.hypo_pose, ambient=0.5, specular=0, shininess=100,
                           light_col=[1, 1, 1], light=[0, 0, -1])
            render_col, _ = ren.finish()
            render_col = render_col.copy()

            cv2.imshow("Input Image", col)

            # Draw FPS in top left corner
            fps = "FPS: " + str(int(1 / (end - start)))

            cv2.rectangle(render_col, (0, 0), (133, 40), (1., 1., 1.), -1)
            cv2.putText(render_col, fps, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow("Refined Output", render_col)
            cv2.waitKey(500)

        orig_trans_err, orig_angular_err = trans_rot_err(gt_pose, perturbed_pose)
        refined_trans_err, refined_angular_err = trans_rot_err(gt_pose, refinable.hypo_pose)

        print('Original Errors')
        print('Translation: {:.4f}m\tRotation: {:.4f}°\n'.format(orig_trans_err, orig_angular_err))

        print('Refined Errors')
        print('Translation: {:.4f}m\tRotation: {:.4f}°'.format(refined_trans_err, refined_angular_err))
