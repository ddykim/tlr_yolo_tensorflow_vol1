import numpy as np
import tensorflow as tf
import yolo.config as cfg
import os
from scipy.io import loadmat

class YOLONet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA
        
        self.weight_path = os.path.join(cfg.DATA_PATH, 'weights', 'yolo_ckpt.mat') # None
        self.debug = False
        
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.logits = self.build_network(self.images, num_outputs=self.output_size, alpha=self.alpha, is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            conv_layer1 = self.create_conv_layer(images, 7, 7, 64, 2, 1, scope='conv_1')            
            maxpool_layer2 = self.create_maxpool_layer(conv_layer1, 2, 2, 2, scope='pool_2')            
            conv_layer3 = self.create_conv_layer(maxpool_layer2, 3, 3, 192, 1, 2, scope='conv_3')            
            maxpool_layer4 = self.create_maxpool_layer(conv_layer3, 2, 2, 2, scope='pool_4')            
            conv_layer5 = self.create_conv_layer(maxpool_layer4, 1, 1, 128, 1, 3, scope='conv_5')
            conv_layer6 = self.create_conv_layer(conv_layer5, 3, 3, 256, 1, 4, scope='conv_6')
            conv_layer7 = self.create_conv_layer(conv_layer6, 1, 1, 256, 1, 5, scope='conv_7')
            conv_layer8 = self.create_conv_layer(conv_layer7, 3, 3, 512, 1, 6, scope='conv_8')            
            maxpool_layer9 = self.create_maxpool_layer(conv_layer8, 2, 2, 2, scope='pool_9')            
            conv_layer10 = self.create_conv_layer(maxpool_layer9, 1, 1, 256, 1, 7, scope='conv_10')
            conv_layer11 = self.create_conv_layer(conv_layer10, 3, 3, 512, 1, 8, scope='conv_11')
            conv_layer12 = self.create_conv_layer(conv_layer11, 1, 1, 256, 1, 9, scope='conv_12')
            conv_layer13 = self.create_conv_layer(conv_layer12, 3, 3, 512, 1, 10, scope='conv_13')
            conv_layer14 = self.create_conv_layer(conv_layer13, 1, 1, 256, 1, 11, scope='conv_14')
            conv_layer15 = self.create_conv_layer(conv_layer14, 3, 3, 512, 1, 12, scope='conv_15')
            conv_layer16 = self.create_conv_layer(conv_layer15, 1, 1, 256, 1, 13, scope='conv_16')
            conv_layer17 = self.create_conv_layer(conv_layer16, 3, 3, 512, 1, 14, scope='conv_17')
            conv_layer18 = self.create_conv_layer(conv_layer17, 1, 1, 512, 1, 15, scope='conv_18')
            conv_layer19 = self.create_conv_layer(conv_layer18, 3, 3, 1024, 1, 16, scope='conv_19')            
            maxpool_layer20 = self.create_maxpool_layer(conv_layer19, 2, 2, 2, scope='pool_20')          
            conv_layer21 = self.create_conv_layer(maxpool_layer20, 1, 1, 512, 1, 17, scope='conv_21')
            conv_layer22 = self.create_conv_layer(conv_layer21, 3, 3, 1024, 1, 18, scope='conv_22')
            conv_layer23 = self.create_conv_layer(conv_layer22, 1, 1, 512, 1, 19, scope='conv_23')
            conv_layer24 = self.create_conv_layer(conv_layer23, 3, 3, 1024, 1, 20, scope='conv_24')
            conv_layer25 = self.create_conv_layer(conv_layer24, 3, 3, 1024, 1, 21, scope='conv_25')
            conv_layer26 = self.create_conv_layer(conv_layer25, 3, 3, 1024, 2, 22, scope='conv_26')
            conv_layer27 = self.create_conv_layer(conv_layer26, 3, 3, 1024, 1, 23, scope='conv_27')
            conv_layer28 = self.create_conv_layer(conv_layer27, 3, 3, 1024, 1, 24, scope='conv_28')
            # flatten layer for connection to fully connected layer
            conv_layer28_flatten_dim = int(reduce(lambda a, b: a * b, conv_layer28.get_shape()[1:]))
            conv_layer28_flatten = tf.reshape(tf.transpose(conv_layer28, (0, 3, 1, 2)), [-1, conv_layer28_flatten_dim])
            connected_layer29 = self.create_connected_layer(conv_layer28_flatten, 512, True, 1, scope='fc_29')
            connected_layer30 = self.create_connected_layer(connected_layer29, 4096, True, 2, scope='fc_30')
            # dropout layer is only used during training
            dropout_layer31= self.create_dropout_layer(connected_layer30, keep_prob, scope='dropout_31')
            connected_layer32= self.create_connected_layer(dropout_layer31, num_outputs, False, 3, scope='fc_32')
            net = connected_layer32
            
            #tf.Print(net)
            
        return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxs1 square and boxs2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(predicts[:, :self.boundary1], [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(predicts[:, self.boundary2:], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            response = tf.reshape(labels[:, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(labels[:, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[:, :, :, 5:]

            offset = tf.constant(self.offset, dtype=tf.float32)
            offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                           (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
                                           tf.square(predict_boxes[:, :, :, :, 2]),
                                           tf.square(predict_boxes[:, :, :, :, 3])])
            predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                                   boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                                   tf.sqrt(boxes[:, :, :, :, 2]),
                                   tf.sqrt(boxes[:, :, :, :, 3])])
            boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)

    def create_conv_layer(self, input_layer, d0, d1, filters, stride, weight_index, scope):
        with tf.variable_scope(scope):
            channels = int(input_layer.get_shape()[3])
            weight_shape = [d0, d1, channels, filters]
            bias_shape = [filters]

            weight = tf.random_normal(weight_shape, stddev = 0.01, dtype = tf.float32)
            bias = tf.random_normal(bias_shape, stddev = 0.01, dtype = tf.float32)
            if self.weight_path:
                variables_mat = loadmat(self.weight_path)
                weight = np.empty(weight_shape, dtype = np.float32)
                weight_what = os.path.join('conv_weight_layer' + str(weight_index))
                weight = variables_mat[weight_what]
                bias = np.empty(bias_shape, dtype = 'float32')
                bias_what = os.path.join('conv_bias_layer' + str(weight_index))
                bias = variables_mat[bias_what]
            
            weight = tf.Variable(weight)
            bias = tf.Variable(bias)
            if self.debug:
                input_layer = tf.Print(input_layer, [input_layer, weight, bias], "convolution")

            # mimic explicit padding used by darknet...a bit tricky
            # https://github.com/pjreddie/darknet/blob/c6afc7ff1499fbbe64069e1843d7929bd7ae2eaa/src/parser.c#L145
            # note that padding integer in yolo-small.cfg actually refers to a boolean value (NOT an acutal padding size)
            d0_pad = int(d0/2)
            d1_pad = int(d1/2)
            input_layer_padded = tf.pad(input_layer, paddings = [[0, 0], [d0_pad, d0_pad], [d1_pad, d1_pad], [0, 0]])
            # we need VALID padding here to match the sizing calculation for output of convolutional used by darknet
            convolution = tf.nn.conv2d(input = input_layer_padded, filter = weight, strides = [1, stride, stride, 1], padding='VALID')
            convolution_bias = tf.add(convolution, bias)
        return self.activation(convolution_bias)

    def create_connected_layer(self, input_layer, d0, leaky, weight_index, scope):
        with tf.variable_scope(scope):
            weight_shape = [int(input_layer.get_shape()[1]), d0]
            bias_shape = [d0]

            weight = tf.random_normal(weight_shape, stddev = 0.01, dtype = tf.float32)
            bias = tf.random_normal(bias_shape, stddev = 0.01, dtype = tf.float32)
            if False:#self.weight_path and not weight_index == 3:
                variables_mat = loadmat(self.weight_path)
                weight = np.empty(weight_shape, dtype = np.float32)
                weight_what= os.path.join('connect_weight_layer' + str(weight_index))
                weight = variables_mat[weight_what]
                bias = np.empty(bias_shape, dtype = 'float32')
                bias_what = os.path.join('connect_bias_layer' + str(weight_index))
                bias = variables_mat[bias_what]

            weight = tf.Variable(weight)
            bias = tf.Variable(bias)
            if self.debug:
                input_layer = tf.Print(input_layer, [input_layer, weight, bias], 'connected')

        return self.activation(tf.add(tf.matmul(input_layer, weight), bias), leaky)

    def create_maxpool_layer(self, input_layer, d0, d1, stride, scope):
        with tf.variable_scope(scope):
            if self.debug:
                input_layer = tf.Print(input_layer, [input_layer], 'pool')
        return tf.nn.max_pool(input_layer, ksize = [1, d0, d1, 1], strides = [1, stride, stride, 1], padding = 'SAME')

    def create_dropout_layer(self, input_layer, prob, scope):
        with tf.variable_scope(scope):
            if self.debug:
                input_layer = tf.Print(input_layer, [input_layer], 'dropout')
        return tf.nn.dropout(input_layer, prob)

    def activation(self, input_layer, leaky = True):
        if leaky:
            # trick to create leaky activation function
            # phi(x) = x if x > 0, 0.1x otherwise
            return tf.maximum(input_layer, tf.scalar_mul(0.1, input_layer))
        else:
            return input_layer
