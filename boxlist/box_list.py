#code_12 BoxList object
import tensorflow as tf

class BoxList(object):
    def __init__(self, boxes):
        """
            boxes: a tensor of shape [N, 4] representing box corners
        """
        if(isinstance(boxes,type(self))):
                boxes=boxes.get()
        if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 5:
            raise ValueError('Invalid dimensions for box data: {}'.format(
                    boxes.shape))
        if boxes.dtype != tf.float32:
            raise ValueError('Invalid tensor type: should be tf.float32')
        self.data = boxes

    def num_boxes(self):

        return tf.shape(self.data)[0]

    def num_boxes_static(self):

        return int(self.data.get_shape()[0]) #shape_utils.get_dim_as_int(self.data['boxes'].get_shape()[0])

    def get(self):

         return self.data

    def set(self, boxes):
        if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 5:
            raise ValueError('Invalid dimensions for box data.')
        self.data = boxes

    def get_center_size_type_box(self, scope=None):

        with tf.name_scope('get_center_size_type_box'):
            box_corners = self.get()
            ymin, xmin, ymax, xmax, angle = tf.unstack(tf.transpose(box_corners))
            width = xmax - xmin
            height = ymax - ymin
            ycenter = ymin + height / 2.
            xcenter = xmin + width / 2.
            return [ycenter, xcenter, height, width, angle]
