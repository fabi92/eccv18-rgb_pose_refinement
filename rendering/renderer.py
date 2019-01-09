

import numpy as np
from vispy import app, gloo

import OpenGL.GL as gl

app.use_app('pyglet')   # Set backend

_vertex_code_colored = """
uniform mat4 u_mv;
uniform mat4 u_mvp;

attribute vec3 a_position;
attribute vec3 a_color;

varying vec3 v_color;
varying vec3 v_V;
varying vec3 v_P;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_color = a_color;
    v_P = gl_Position.xyz; // v_P is the world position
    v_V = (u_mv * vec4(a_position, 1.0)).xyz;
}
"""

_fragment_code_bbox = """
varying vec3 v_color;
varying vec3 v_V;
varying vec3 v_P;

void main() {
    gl_FragColor = vec4(v_color, 1.0);
}
"""

_fragment_code_colored = """
uniform float u_ambient;
uniform float u_specular;
uniform float u_shininess;
uniform vec3 u_light_dir;
uniform vec3 u_light_col;

varying vec3 v_color;
varying vec3 v_V;
varying vec3 v_P;

void main() {
    vec3 N = normalize(cross(dFdy(v_P), dFdx(v_P))); // N is the world normal
    vec3 V = normalize(v_V);
    vec3 R = reflect(V, N);
    vec3 L = normalize(u_light_dir);

    vec3 ambient = v_color * u_light_col * u_ambient;
    vec3 diffuse = v_color * u_light_col * max(dot(L, N), 0.0);
    float specular = u_specular * pow(max(dot(R, L), 0.0), u_shininess);
    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
}
"""

_vertex_code_textured = """
uniform mat4 u_mv;
uniform mat4 u_mvp;

attribute vec3 a_position;
attribute vec2 a_texcoord;

varying vec2 v_texcoord;
varying vec3 v_V;
varying vec3 v_P;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_texcoord = a_texcoord;
    v_P = gl_Position.xyz; // v_P is the world position
    v_V = (u_mv * vec4(a_position, 1.0)).xyz;
}
"""

_fragment_code_textured = """
uniform float u_ambient;
uniform float u_specular;
uniform float u_shininess;
uniform vec3 u_light_dir;
uniform vec3 u_light_col;

uniform sampler2D u_tex;

varying vec2 v_texcoord;
varying vec3 v_V;
varying vec3 v_P;

void main() {
    vec3 N = normalize(cross(dFdy(v_P), dFdx(v_P))); // N is the world normal
    vec3 V = normalize(v_V);
    vec3 R = reflect(V, N);
    vec3 L = normalize(u_light_dir);

    vec3 color = texture2D(u_tex, v_texcoord).xyz;
    vec3 ambient = color * u_light_col * u_ambient;
    vec3 diffuse = color * u_light_col * max(dot(L, N), 0.0);
    float specular = u_specular * pow(max(dot(R, L), 0.0), u_shininess);
    gl_FragColor = vec4(ambient + diffuse + specular, 1.0);

}
"""

_vertex_code_background = """

attribute vec2 a_position;
attribute vec2 a_texcoord;

varying vec2 v_texcoord;

void main() {
    gl_Position = vec4(a_position, 0, 1.0);
    v_texcoord = a_texcoord;
}
"""

_fragment_code_background = """
uniform sampler2D u_tex;

varying vec2 v_texcoord;

void main() {
    gl_FragColor = texture2D(u_tex, v_texcoord);
}
"""


def singleton(cls):
    instances = {}

    def get_instance(size, cam):
        if cls not in instances:
            instances[cls] = cls(size, cam)
        return instances[cls]
    return get_instance


@singleton  # Don't throw GL context into trash when having more than one Renderer instance
class Renderer(app.Canvas):

    def __init__(self, size, cam):

        app.Canvas.__init__(self, show=False, size=size)
        self.shape = (size[1], size[0])
        self.yz_flip = np.eye(4, dtype=np.float32)
        self.yz_flip[1, 1], self.yz_flip[2, 2] = -1, -1

        self.set_cam(cam)

        # Set up shader programs
        self.program_col = gloo.Program(_vertex_code_colored, _fragment_code_colored)
        self.program_bbox = gloo.Program(_vertex_code_colored, _fragment_code_bbox)
        self.program_tex = gloo.Program(_vertex_code_textured, _fragment_code_textured)
        self.program_bg = gloo.Program(_vertex_code_background, _fragment_code_background)

        # Texture where we render the color/depth and its FBO
        self.col_tex = gloo.Texture2D(shape=self.shape + (3,))
        self.fbo = gloo.FrameBuffer(self.col_tex, gloo.RenderBuffer(self.shape))
        self.fbo.activate()
        gloo.set_state(depth_test=True, blend=False, cull_face=True)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gloo.set_clear_color((0.0, 0.0, 0.0))
        gloo.set_viewport(0, 0, *self.size)

        # Set up background render quad in NDC
        quad = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
        tex = [[0, 1], [1, 1], [1, 0], [0, 0]]
        vertices_type = [('a_position', np.float32, 2), ('a_texcoord', np.float32, 2)]
        collated = np.asarray(list(zip(quad, tex)), vertices_type)
        self.bg_vbuffer = gloo.VertexBuffer(collated)
        self.bg_ibuffer = gloo.IndexBuffer([0, 1, 2, 0, 2, 3])

    def set_cam(self, cam, clip_near=0.01, clip_far=10.0):
        self.cam = cam
        self.clip_near = clip_near
        self.clip_far = clip_far
        self.mat_proj = self.projective_matrix(cam, 0, 0,
                                               self.shape[1], self.shape[0],
                                               clip_near, clip_far)
    def clear(self, color=True, depth=True):
        gloo.clear(color=color, depth=True)

    def finish(self):

        im = gl.glReadPixels(0, 0, self.size[0], self.size[1], gl.GL_RGB, gl.GL_FLOAT)
        rgb = np.copy(np.frombuffer(im, np.float32)).reshape(self.shape+(3,))[::-1, :]  # Read buffer and flip Y
        im = gl.glReadPixels(0, 0, self.size[0], self.size[1], gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        dep = np.copy(np.frombuffer(im, np.float32)).reshape(self.shape+(1,))[::-1, :]  # Read buffer and flip Y

        # Convert z-buffer to depth map
        mult = (self.clip_near*self.clip_far)/(self.clip_near-self.clip_far)
        addi = self.clip_far/(self.clip_near-self.clip_far)
        bg = dep == 1
        dep = mult/(dep + addi)
        dep[bg] = 0
        return rgb, np.squeeze(dep)

    def draw_background(self, image):
        self.program_bg['u_tex'] = gloo.Texture2D(image)
        self.program_bg.bind(self.bg_vbuffer)
        self.program_bg.draw('triangles', self.bg_ibuffer)
        gloo.clear(color=False, depth=True)  # Clear depth


    def draw_model(self, model, pose, ambient=0.15, specular=0.1, shininess=0.1,
                   light=(0, 0, 1), light_col=(1, 1, 1)):

        # View matrix (transforming the coordinate system from OpenCV to OpenGL camera space)
        mv = (self.yz_flip.dot(pose)).T    # OpenCV to OpenGL camera system (flipped, column-wise)
        mvp = mv.dot(self.mat_proj)

        used_program = self.program_col
        if model.texcoord is not None:
            used_program = self.program_tex
            used_program['u_tex'] = model.texture

        used_program.bind(model.vertex_buffer)
        used_program['u_light_dir'] = light
        used_program['u_light_col'] = light_col
        used_program['u_ambient'] = ambient
        used_program['u_specular'] = specular
        used_program['u_shininess'] = shininess
        used_program['u_mv'] = mv
        used_program['u_mvp'] = mvp
        used_program.draw('triangles', model.index_buffer)

    def draw_boundingbox(self, model, pose, thickness=3.):

        # View matrix (transforming the coordinate system from OpenCV to OpenGL camera space)
        mv = (self.yz_flip.dot(pose)).T  # OpenCV to OpenGL camera system (flipped, column-wise)
        mvp = mv.dot(self.mat_proj)

        self.program_bbox.bind(model.bb_vbuffer)
        self.program_bbox['u_mv'] = mv
        self.program_bbox['u_mvp'] = mvp
        gloo.set_line_width(thickness)
        self.program_bbox.draw('lines', model.bb_ibuffer)

    def draw_coordinate_system(self, model, pose):
        self.clear(color=False)

        # View matrix (transforming the coordinate system from OpenCV to OpenGL camera space)
        mv = (self.yz_flip.dot(pose)).T  # OpenCV to OpenGL camera system (flipped, column-wise)
        mvp = mv.dot(self.mat_proj)

        self.program_bbox.bind(model.cs_vbuffer)
        self.program_bbox['u_mv'] = mv
        self.program_bbox['u_mvp'] = mvp
        gloo.set_line_width(width=3.0)
        self.program_bbox.draw('lines', model.cs_ibuffer)
        gloo.set_line_width(width=1.0)

    def projective_matrix(self, cam, x0, y0, w, h, nc, fc):

        q = -(fc + nc) / float(fc - nc)
        qn = -2 * (fc * nc) / float(fc - nc)

        # Draw our images upside down, so that all the pixel-based coordinate systems are the same
        proj = np.array([
            [2 * cam[0, 0] / w, -2 * cam[0, 1] / w, (-2 * cam[0, 2] + w + 2 * x0) / w, 0],
            [0, -2 * cam[1, 1] / h, (-2 * cam[1, 2] + h + 2 * y0) / h, 0],
            [0, 0, q, qn],  # This row is standard glPerspective and sets near and far planes
            [0, 0, -1, 0]
        ])

        # Compensate for the flipped image
        proj[1, :] *= -1.0
        return proj.T

    def compute_metrical_clip(self, pose, diameter):

        width = self.cam[0, 0] * diameter / pose[2, 3]  # X coordinate == shape[1]
        height = self.cam[1, 1] * diameter / pose[2, 3]  # Y coordinate == shape[0]
        proj = np.matmul(self.cam, pose[0:3, 3])
        proj /= proj[2]
        cut = np.asarray([proj[1] - height//2, proj[0] - width//2, proj[1] + height//2, proj[0] + width//2], dtype=int)

        # Can lead to offsetted extractions, not really nice...
        cut[0] = np.clip(cut[0], 0, self.shape[0])
        cut[2] = np.clip(cut[2], 0, self.shape[0])
        cut[1] = np.clip(cut[1], 0, self.shape[1])
        cut[3] = np.clip(cut[3], 0, self.shape[1])
        return cut

    def render_view_metrical_clip(self, model, pose, diameter):

        cut = self.compute_metrical_clip(pose, diameter)
        self.clear()
        self.draw_model(model, pose)
        col, dep = self.finish()
        return col[cut[0]:cut[2], cut[1]:cut[3]], dep[cut[0]:cut[2], cut[1]:cut[3]]
