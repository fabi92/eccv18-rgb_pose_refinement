import cv2
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.distance import cdist
from utils.quaternion import matrix2quaternion

def backproject(depth, cam):
    """ Backproject a depth map to a cloud map
    :param depth: Input depth map
    :param cam: Intrinsics of the camera
    :return: An organized cloud map
    """
    X = np.asarray(range(depth.shape[1])) - cam[0, 2]
    X = np.tile(X, (depth.shape[0], 1))
    Y = np.asarray(range(depth.shape[0])) - cam[1, 2]
    Y = np.tile(Y, (depth.shape[1], 1)).transpose()
    return np.stack((X * depth / cam[0, 0], Y * depth / cam[1, 1], depth), axis=2)


def project(points_3d, cam):
    """ Project a numpy array of 3D points to the image plane
    :param points_3d: Input array of 3D points (N, 3)
    :param cam: Intrinsics of the camera
    :return: An array of projected 2D points
    """
    x = cam[0, 2] + points_3d[:, 0] * cam[0, 0] / points_3d[:, 2]
    y = cam[1, 2] + points_3d[:, 1] * cam[1, 1] / points_3d[:, 2]
    return np.stack((x, y), axis=1).astype(np.uint32)


def depthTo3D(pt, z, cam):
    """ Backproject a single 2D point to a 3D point
        :param pt: Input 2D point
        :param z: Inputh depth value of 2D point
        :param cam: Intrinsics of the camera
        :return: Computed 3D scene point
        """
    assert len(pt) == 2
    x = pt[0]
    y = pt[1]

    fx = cam[0, 0]
    fy = cam[1, 1]
    ox = cam[0, 2]
    oy = cam[1, 2]

    inv_fx = 1. / fx
    inv_fy = 1. / fy
    return np.array([(x - ox) * inv_fx * z, (y - oy) * inv_fy * z, z])


def get_viewpoint_cloud(depth, cam, num_keep):
    """ Extract 3d points from depth map and intrinsics
    :param depth:
    :param cam:
    :return: Numpy array of 3D points (N, 3)
    """
    dep_c = depth.copy()
    cloud = backproject(depth, cam)
    contours_from_dep = extract_contour(depth)

    dep_c[contours_from_dep == 0] = 0
    mask = np.stack((dep_c, dep_c, dep_c), axis=2) > 0
    contours = np.reshape(cloud[mask], (-1, 3))
    contours = contours[np.random.choice(
        contours.shape[0], num_keep)]

    return contours


def get_full_viewpoint_cloud(depth, cam, num_keep):
    """ Extract 3d points from depth map and intrinsics
    :param depth:
    :param cam:
    :return: Numpy array of 3D points (N, 3)
    """
    dep_c = depth.copy()
    cloud = backproject(depth, cam)
    mask = np.stack((dep_c, dep_c, dep_c), axis=2) > 0
    cloud = np.reshape(cloud[mask], (-1, 3))
    cloud = cloud[np.random.choice(
        cloud.shape[0], num_keep)]

    return cloud

def extract_contour(input):
    """ Extracts contour where input jumps from 0 to another value
    :param input:
    :return: A binary contour map in np.float32
    """
    mask = (input > 0).astype(np.float32)
    return mask*(1-cv2.erode(mask, np.ones((3, 3), np.uint8)))


def heatmap(input):
    """ Returns a RGB heatmap representation
    :param input:
    :return:
    """
    min, max = np.amin(input), np.amax(input)
    rescaled = 255*((input-min)/(max-min))
    return cv2.applyColorMap(rescaled.astype(np.uint8), cv2.COLORMAP_JET)


def distance_transform(depth):
    """ Returns a distance transform for a depth map.
    :param depth: Zero values are exterior, non-zero values are interior area
    :return: The distance transform, signed and unsigned
    """
    mask = (depth > 0).astype(np.float32)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
    contours = mask*(1-eroded)
    dt_unsigned = cv2.distanceTransform((1-contours).astype(np.uint8), cv2.DIST_L2, 3)
    dt_signed = np.copy(dt_unsigned)
    dt_signed[eroded.astype(bool)] = -dt_signed[eroded.astype(bool)]
    return dt_unsigned, dt_signed


def transform_points(points_3d, mat):
    """ Apply rigid body motion to an array of 3D points
    :param points_3d: Numpy array of 3D points (N, 3)
    :param mat: 4x4 matrix of the transform
    :return: The transformed points array
    """
    rot = np.matmul(mat[:3, :3], points_3d.transpose())
    return rot.transpose() + mat[:3, 3]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    return interArea / float(boxAArea + boxBArea - interArea)

def get_occlusion(depA, depB):
    interesection = np.sum(depA * depB)
    gt_area = np.sum(depA)

    return float(interesection) / float(gt_area)

def vss(ren, model, poseA, poseB):
    ren.clear()
    ren.draw_model(model, poseA)
    _, depA = ren.finish()
    depA = depA > 0

    ren.clear()
    ren.draw_model(model, poseB)
    _, depB = ren.finish()
    depB = depB > 0

    interesection = np.sum(depA * depB)
    union = np.sum(depA + depB)

    return float(interesection) / float(union)

def add(poseA, poseB, model):
    v_A = transform_points(model.vertices, poseA)
    v_B = transform_points(model.vertices, poseB)

    return np.mean(np.linalg.norm(v_A - v_B, axis=1))


def adi(poseA, poseB, model):
    v_A = transform_points(model.vertices, poseA)
    v_B = transform_points(model.vertices, poseB)
    dist = cdist(v_A, v_B)

    return np.mean(np.min(dist, axis=1))


def verify_objects_in_scene(dep):
    sliced = np.argwhere(dep)
    if sliced.size == 0:
        return None

    # BBox is in form xmin, ymin, xmax, ymax
    bbox = [sliced.min(0)[1],
            sliced.min(0)[0],
            sliced.max(0)[1] + 1,
            sliced.max(0)[0] + 1]

    centroid = (bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2)
    if centroid[0] < 0 or centroid[1] < 0 or centroid[0] > 640 or centroid[1] > 480:
        return None

    centroid = [int(bbox[0] + (bbox[2] - bbox[0]) / 2), int(bbox[1] + (bbox[3] - bbox[1]) / 2)]

    return centroid


def perturb_pose(pose, rot_variation, trans_variation):

    rodriguez = 2 * np.random.uniform(0, 1, 3) - 1
    rodriguez /= np.linalg.norm(rodriguez)
    perturb_quat = Quaternion(axis=rodriguez,
                              angle=np.random.uniform(0, 1) * 2 * rot_variation - rot_variation)  #

    if perturb_quat[0] < 0:
        perturb_quat *= -1.

    vector = np.random.uniform(-1, 1, 3)
    vector = vector / np.linalg.norm(vector)

    perturb_tra = vector * np.random.uniform(0, trans_variation)

    hy_pose = np.identity(4)
    hy_pose[:3, :3] = np.dot(perturb_quat.rotation_matrix, pose[:3, :3])
    hy_pose[:3, 3] = pose[:3, 3] + perturb_tra

    return hy_pose


def trans_rot_err(poseA, poseB):
    last_trans = poseA[:3, 3]
    last_rot = Quaternion(matrix2quaternion(poseA[:3, :3]))

    cur_trans = poseB[:3, 3]
    cur_rot = Quaternion(matrix2quaternion(poseB[:3, :3]))

    trans_diff = np.linalg.norm(cur_trans - last_trans)
    update_q = cur_rot * last_rot.inverse
    angular_diff = np.abs((update_q).degrees)

    return trans_diff, angular_diff