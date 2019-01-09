import cv2
import numpy as np

from rendering.utils import verify_objects_in_scene
from utils.quaternion import matrix2quaternion
from pyquaternion import Quaternion


class Refinable(object):
    def __init__(self, model, label, metric_crop_shape, input_col=None, hypo_pose=np.identity(4)):
        self.metric_crop_shape = metric_crop_shape
        self.label = label
        self.model = model
        self.input_col = input_col
        self.hypo_pose = hypo_pose
        self.bbox = None
        self.hypo_dep = None


class Refiner(object):

    def __init__(self, architecture, ren, session):
        self.ren = ren
        self.architecture = architecture
        self.session = session

    def iterative_contour_alignment(self, refinable, max_iterations=3,
                                    min_rotation_displacement=0.5,
                                    min_translation_displacement=0.0025, display=False):
        assert refinable is not None

        last_pose = np.copy(refinable.hypo_pose)
        for i in range(max_iterations):

            refinable = self.refine(refinable=refinable)

            last_trans = last_pose[:3, 3]
            last_rot = Quaternion(matrix2quaternion(last_pose[:3, :3]))

            cur_trans = refinable.hypo_pose[:3, 3]
            cur_rot = Quaternion(matrix2quaternion(refinable.hypo_pose[:3, :3]))

            trans_diff = np.linalg.norm(cur_trans - last_trans)
            update_q = cur_rot * last_rot.inverse
            angular_diff = np.abs((update_q).degrees)

            last_pose = np.copy(refinable.hypo_pose)

            if display:
                concat = cv2.hconcat([refinable.input_col, refinable.hypo_col])
                cv2.imshow('test', concat)
                cv2.waitKey(500)

            if angular_diff <= min_rotation_displacement and trans_diff <= min_translation_displacement:
                refinable.iterations = i+1
                return refinable

        refinable.iterations = max_iterations
        return refinable

    def refine(self, refinable):

        refinable.refined = False
        self.ren.clear()
        self.ren.draw_model(refinable.model, refinable.hypo_pose, ambient=0.5, specular=0, shininess=100,
                            light_col=[1, 1, 1], light=[0, 0, -1])
        refinable.hypo_col, refinable.hypo_dep = self.ren.finish()

        # padding to prevent crash when object gets to close to border
        pad = int(refinable.metric_crop_shape[0] / 2)
        input_col = np.pad(refinable.input_col, ((pad, pad), (pad, pad), (0, 0)),'wrap')
        hypo_col = np.pad(refinable.hypo_col, ((pad, pad), (pad, pad), (0, 0)), 'wrap')

        centroid = verify_objects_in_scene(refinable.hypo_dep)

        if centroid is None:
            print("Hypo outside of image plane")
            return refinable

        (x, y) = centroid
        x_normalized = x / 640.
        y_normalized = y / 480.

        # crop to metric shape
        slice = (int(refinable.metric_crop_shape[0] / 2), int(refinable.metric_crop_shape[1] / 2))
        input_col = input_col[y: y + 2 * slice[1], x: x + 2 * slice[0]]
        hypo_col = hypo_col[y: y + 2 * slice[1], x: x + 2 * slice[0]]
        input_shape = (self.architecture.input_shape[0], self.architecture.input_shape[1])

        # resize to input shape of architecture
        scene_patch = cv2.resize(input_col, input_shape)
        render_path = cv2.resize(hypo_col, input_shape)

        # write feed dict
        hypo_trans = refinable.hypo_pose[:3, 3]
        hypo_rot = matrix2quaternion(refinable.hypo_pose[:3, :3])
        if hypo_rot[0] < 0.:
            hypo_rot *= -1

        feed_dict = {
            self.architecture.scene_patch: [scene_patch],
            self.architecture.render_patch: [render_path],
            self.architecture.hypo_rotation: hypo_rot.reshape(1, 4),
            self.architecture.hypo_translation: hypo_trans.reshape(1, 3),
            self.architecture.crop_shift: [[x_normalized, y_normalized]]
        }

        # run network
        refined_rotation, refined_translation = self.session.run([self.architecture.rotation_hy_to_gt,
                                                                  self.architecture.translation_hy_to_gt],
                                                                 feed_dict=feed_dict)

        assert np.sum(np.isnan(refined_translation[0])) == 0 and np.sum(np.isnan(refined_rotation[0])) == 0

        refined_pose = np.identity(4)
        refined_pose[:3, :3] = Quaternion(refined_rotation[0]).rotation_matrix
        refined_pose[:3, 3] = refined_translation[0]

        refinable.hypo_pose = refined_pose
        refinable.render_patch = render_path.copy()
        refinable.refined = True

        return refinable
