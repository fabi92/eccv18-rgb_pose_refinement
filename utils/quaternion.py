import numpy as np
from pyquaternion import Quaternion
from functools import partial

from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.linalg import expm, logm

def matrix2quaternion(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


def quaternion_to_euler(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = np.degrees(np.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.atan2(t3, t4))

    return [X, Y, Z]

def find_symmetry_quaternions(data):
    xyz = data[:, 1:4]
    xyz = xyz / np.linalg.norm(xyz, axis=1)[:, np.newaxis]
    U, S, V = np.linalg.svd(xyz)

    axis_initializer = V[S.argmin()]

    def loss(A, x):
        return np.linalg.norm(np.dot(A, x), ord=1)

    res = minimize(partial(loss, xyz), axis_initializer, method='nelder-mead', options={'xtol': 1e-8, 'disp': False})

    return res.x / np.linalg.norm(res.x), axis_initializer/np.linalg.norm(axis_initializer)


def validate_symmetry_axis(rotations, lambda0_threshold=1.2, lambda1_threshold=1.):
    pca = PCA(n_components=4)
    pca.fit(rotations / np.linalg.norm(rotations, axis=1)[:, np.newaxis])
    E = pca.singular_values_

    if E[1] > lambda1_threshold:
        return True

    return False


def weiszfeld_interpolation(quaternions, iterations=1000):
    y = Quaternion(quaternions[0].tolist())
    for i in range(iterations):
        newy = Quaternion()
        dividend = 0
        for j in range(1, len(quaternions)):
            distance = Quaternion.distance(Quaternion(quaternions[j]), y)
            if distance < 1e-4:
                continue
            newy = newy + quaternions[j] / distance
            dividend += 1 / distance
        newy /= dividend
        y = newy
    return y


def exp_map(b, p):
    """theta = norm(b-p);
            dminusbx = sqrt(2-2.*cos(pi-theta));
            l = 2.*sin(theta/2);
            alpha = acos( (4+dminusbx.^2-l.^2)./(4*dminusbx) );
            dpb = 2.*tan(alpha);
            v = b + ((p-b)./norm(p-b)).*dpb;
            x = ((v+b)./norm(v+b)).*dminusbx-b;"""
    """
    EXP_MAP The exponential map for n-spheres
    b is the base point (vector in R^n), norm(b)=1
    p is a point on the tangent plane to the hypersphere at b (also a vector in R^n)

    method can be 0 or 1:
    0: hypersphere (e.g. quaternions)
    1: dual quaternion
    """
    if np.allclose(b, p):
        x = b
    else:
        theta = np.linalg.norm(b - p)
        dminusbx = np.sqrt(2 - 2. * np.cos(np.pi - theta))
        l = 2. * np.sin(theta / 2)
        alpha = np.arccos((4 + dminusbx ** 2 - l ** 2) / (4 * dminusbx))
        dpb = 2. * np.tan(alpha)
        v = b + ((p - b) / np.linalg.norm(p - b)) * dpb
        x = ((v + b) / np.linalg.norm(v + b)) * dminusbx - b

    return x


def log_map(b, x):
    """theta = acos(dot(b,x));
            alpha = acos((4+norm(b+x).^2-norm(x-b).^2)./(2.*2.*norm(b+x)));
            p2 = (2.*(b+x))./(norm(b+x).*cos(alpha)) - b;
            p = b+((p2-b)./norm(p2-b)).*theta;"""
    """
    LOG_MAP The log map for n-spheres
    b is the base point (vector in R^n), norm(b)=1
    x is a point on the hypersphere (also a vector in R^n), norm(x)=1

    method can be 0 or 1:
    0: hypersphere (e.g. quaternions)
    1: dual quaternion"""

    if np.allclose(x, b):
        p = b
    else:
        theta = np.arccos(np.dot(b, x))
        alpha = np.arccos(
            (4 + np.linalg.norm(b + x) ** 2 - np.linalg.norm(x - b) ** 2) / (2. * 2. * np.linalg.norm(b + x)))
        p2 = (2. * (b + x)) / (np.linalg.norm(b + x) * np.cos(alpha)) - b
        p = b + ((p2 - b) / np.linalg.norm(p2 - b)) * theta

    return p


def geodesic_mean(quaternions, epsilon=1e-2, max_iterations=1000):
    mu = Quaternion(quaternions[0]).rotation_matrix
    for i in range(max_iterations):
        avgX = np.zeros((3, 3), dtype=np.float32)
        for quat in quaternions:
            dXi = np.linalg.solve(mu, Quaternion(quat).rotation_matrix)
            dxu = logm(dXi)
            avgX = avgX + dxu

        dmu = expm((1. / len(quaternions)) * avgX)
        mu = np.matmul(mu, dmu)
        if np.linalg.norm(logm(dmu)) <= epsilon:
            break

    return Quaternion(matrix2quaternion(mu))