import tensorflow as tf


class Architecture(object):

    def __init__(self, network_file, sess):

        assert sess is not None

        self.load_frozen_graph(network=network_file)

        # input tensors
        self.scene_patch = sess.graph.get_tensor_by_name('input_patches:0')
        self.render_patch = sess.graph.get_tensor_by_name('hypo_patches:0')
        self.hypo_rotation = sess.graph.get_tensor_by_name('hypo_rotations:0')
        self.hypo_translation = sess.graph.get_tensor_by_name('hypo_translations:0')
        self.crop_shift = sess.graph.get_tensor_by_name('cropshift:0')

        # in architecture saved information
        self.input_shape = [224, 224, 3]

        # output tensors
        self.rotation_hy_to_gt = sess.graph.get_tensor_by_name('refined_rotation:0')
        self.translation_hy_to_gt = sess.graph.get_tensor_by_name('refined_translation:0')

    def load_frozen_graph(self, network):
        """ Loads the provided network as the new default graph """
        with tf.gfile.FastGFile(network, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
