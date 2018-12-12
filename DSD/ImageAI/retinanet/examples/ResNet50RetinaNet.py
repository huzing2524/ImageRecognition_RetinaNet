# -*- coding: utf-8 -*-
import os
# import time
import base64
import cv2
import keras
import matplotlib
# import miscellaneous modules
import matplotlib.pyplot as plt
import numpy as np
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption

# 保存图片，不使用matplotlib画图
matplotlib.use("Agg")

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# use this environment flag to change which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join(os.getcwd(), 'ImageAI/retinanet/examples/resnet50_coco_best_v2.1.0.h5')
# print(model_path)
# print(os.getcwd())

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)

# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_chinese = {0: '人', 1: '自行车', 2: '汽车', 3: '摩托车', 4: '飞机', 5: '公共汽车', 6: '火车',
                     7: '卡车', 8: '船', 9: '交通信号灯', 10: '消防栓', 11: '停车标志', 12: '停车计时器',
                     13: '长凳', 14: '鸟', 15: '猫', 16: '狗', 17: '马', 18: '绵羊', 19: '牛', 20: '大象',
                     21: '熊', 22: '斑马', 23: '长颈鹿', 24: '双肩背包', 25: '雨伞', 26: '手提包', 27: '领带',
                     28: '手提箱', 29: '飞盘', 30: '滑雪', 31: '滑雪板', 32: '运动球', 33: '风筝',
                     34: '棒球棒', 35: '棒球手套', 36: '滑板', 37: '冲浪板', 38: '网球拍',
                     39: '瓶子', 40: '酒杯', 41: '杯子', 42: '叉子', 43: '刀', 44: '勺', 45: '碗',
                     46: '香蕉', 47: '苹果', 48: '三明治', 49: '橙子', 50: '西兰花', 51: '胡萝卜', 52: '热狗',
                     53: '比萨', 54: '甜甜圈', 55: '蛋糕', 56: '椅子', 57: '长椅', 58: '盆栽', 59: '床',
                     60: '餐桌', 61: '厕所', 62: '电视', 63: '笔记本电脑', 64: '老鼠', 65: '遥控器', 66: '键盘',
                     67: '手机', 68: '微波炉', 69: '烤箱', 70: '烤面包机', 71: '水槽', 72: '冰箱',
                     73: '书', 74: '时钟', 75: '花瓶', 76: '剪刀', 77: '泰迪熊', 78: '吹风机',
                     79: '牙刷'}

labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                   7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
                   13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                   21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
                   28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                   34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                   39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                   46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                   53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                   60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                   67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                   73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                   79: 'toothbrush'}


def predict(image_id):
    """
    image_id: 请求图片id
    :return: {
              "image": "string",  // 识别结果图像二进制数据的base64编码
              "desc": "string",  // 识别物品的描述
              "time": 0.88      //  识别结果概率
            }
    """
    # load image
    result = {}

    img_path = os.path.join(os.getcwd(), 'ImageAI/retinanet/finished')
    image_path = img_path + '/{}.jpg'.format(image_id)

    image = read_image_bgr(image_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    # start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    # print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # 保留最大概率的一个物体
    box, score, label = boxes[0][0], scores[0][0], labels[0][0]
    # print(box, score, label)  # [4.111461e-01 3.761422e+02 1.161281e+02 4.784541e+02] 0.9396883 2

    color = label_color(label)

    b = box.astype(int)
    draw_box(draw, b, color=color)

    caption = "{} {:.3f}".format(labels_to_names[label], score)  # 'airplane 0.997'
    draw_caption(draw, b, caption)

    # 画图
    plt.figure(figsize=(30, 30))
    plt.axis('off')
    plt.imshow(draw)
    # 显示图片
    # plt.show()
    # 保存图片
    finished_path = os.path.join(os.getcwd(), 'ImageAI/retinanet/finished')
    plt.savefig(finished_path + "/{}.jpg".format(image_id))

    with open(finished_path + "/{}.jpg".format(image_id), "rb") as f:
        img = f.read()

    image = base64.b64encode(img)

    caption_to_chinese = "{} {:.3f}".format(labels_to_chinese[label], score)
    li = caption_to_chinese.split(" ")
    result["image"] = image
    result["desc"] = li[0]
    result["time"] = li[-1]

    return result
