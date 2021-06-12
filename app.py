from flask import Flask, render_template,  request, redirect, flash, url_for
from flask import Flask
import os
import time
import urllib.request

#-----------------------------#
import cv2
import numpy as np
import tensorflow as tf
from absl import logging
from itertools import repeat
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Concatenate, Lambda
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU
from tensorflow.keras.layers import MaxPool2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2
#----------------------------#
from tensorflow.python.keras.backend import shape

app = Flask(__name__)
from werkzeug.utils import secure_filename
app.secret_key = "secret key"
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------------- ROUTE ------------------#
@app.route('/')
def upload_form():
		return render_template('index.html', filename="meo.jpg", output="meo.jpg", name="'Trong ảnh của bạn, chúng tôi tìm thấy: 2preson, 1 cat")   # example image
@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		import cv2
		im = cv2.imread('static/uploads/'+filename)
		cv2.imwrite("static/uploads/rs"+filename,im)
		img = tf.image.decode_image(open("static/uploads/rs"+filename, 'rb').read(), channels=3)
		img = tf.expand_dims(img, 0)  # (x,y,3) > (1,x,y,3)
		img = (tf.image.resize(img, (size, size))) / 255   

		t1 = time.time()  # dem thoi gian
		boxes, scores, classes, nums = yolo(img) #eager mode
		t2 = time.time()  # ket thuc dem (du doan xong)

		imgoutput = cv2.imread("static/uploads/rs"+filename) # load anh goc
		imgoutput = draw_outputs(imgoutput, (boxes, scores, classes, nums), class_names)
		cv2.imwrite("static/outputs/"+filename,imgoutput)
		print("----------------------------")
		kq=[]
		print('time: {}'.format(t2 - t1))
		print('detections:')
		print(nums)
		for i in range(nums[0]):
			print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
												np.array(scores[0][i]),
												np.array(boxes[0][i])))
			kq.append(class_names[int(classes[0][i])])
		showname= 'Trong ảnh của bạn, chúng tôi tìm thấy: '
		from collections import Counter
		newList = Counter(kq)
		checkDup=[]
		for i in kq:
			if not i in checkDup:
				showname=showname+ str(newList[i]) + i + ", "
				checkDup.append(i)
		
		return render_template('index.html', filename=filename, output=filename, name=showname)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display2/<filename>')	
def display_output(filename):
	return redirect(url_for('static', filename='outputs/' + filename), code=301)
# ---------------------end: ROUTE ------------------#

# ----------------------  DEFINE SOMETHING------------------#
def initdetect(): 
	a=1
	# load_darknet_weights(yolo, weightsyolov3)
	# yolo.save_weights(checkpoints)
#Define some imporant value which we will use later
yolo_iou_threshold   = 0.6 # iou threshold
yolo_score_threshold = 0.6 # score threshold
# Edit model hể: (cau truc)
weightsyolov3 = 'yolov3.weights' #  weights file
weights= 'checkpoints/yolov3.tf' #  checkpoints file
size= 416             #resize images to\
checkpoints = 'checkpoints/yolov3.tf'
num_classes = 80      # nums of classes in the model

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  #Defining 3 anchor boxes for each grid
                        (59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

class_names =  ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
	"banana","apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake","chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
	"mouse","remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
	"refrigerator","book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
YOLO_V3_LAYERS = [
  'yolo_darknet',
  'yolo_conv_0',
  'yolo_output_0',
  'yolo_conv_1',
  'yolo_output_1',
  'yolo_conv_2',
  'yolo_output_2',
]

class BatchNormalization(tf.keras.layers.BatchNormalization):
  def call(self, x, training=False):
    if training is None: training = tf.constant(False)
    training = tf.logical_and(training, self.trainable)
    return super().call(x, training)

# Create our model(.weights >>> .tf), load weights and class names. There is 80 of them in Coco dataset. 
def YoloV3(size=None, channels=3, anchors=yolo_anchors,
          masks=yolo_anchor_masks, classes=80, training=False):
  x = inputs = Input([size, size, channels])

  x_36, x_61, x = Darknet(name='yolo_darknet')(x)

  x = YoloConv(512, name='yolo_conv_0')(x)
  output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

  x = YoloConv(256, name='yolo_conv_1')((x, x_61))
  output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

  x = YoloConv(128, name='yolo_conv_2')((x, x_36))
  output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

  if training:
    return Model(inputs, (output_0, output_1, output_2), name='yolov3')

  boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                  name='yolo_boxes_0')(output_0)
  boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                  name='yolo_boxes_1')(output_1)
  boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                  name='yolo_boxes_2')(output_2)

  outputs = Lambda(lambda x: nonMaximumSuppression(x, anchors, masks, classes),
                  name='nonMaximumSuppression')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

  return Model(inputs, outputs, name='yolov3')

def yolo_boxes(pred, anchors, classes):
	grid_size = tf.shape(pred)[1]
	box_xy, box_wh, score, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

	box_xy = tf.sigmoid(box_xy)
	score = tf.sigmoid(score)
	class_probs = tf.sigmoid(class_probs)
	pred_box = tf.concat((box_xy, box_wh), axis=-1)

	grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
	grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

	box_xy = (box_xy + tf.cast(grid, tf.float32)) /  tf.cast(grid_size, tf.float32)
	box_wh = tf.exp(box_wh) * anchors

	box_x1y1 = box_xy - box_wh / 2
	box_x2y2 = box_xy + box_wh / 2
	bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

	return bbox, score, class_probs, pred_box

#Non-maximum suppression
def nonMaximumSuppression(outputs, anchors, masks, classes):
  boxes, conf, out_type = [], [], []

  for output in outputs:
    boxes.append(tf.reshape(output[0], (tf.shape(output[0])[0], -1, tf.shape(output[0])[-1])))
    conf.append(tf.reshape(output[1], (tf.shape(output[1])[0], -1, tf.shape(output[1])[-1])))
    out_type.append(tf.reshape(output[2], (tf.shape(output[2])[0], -1, tf.shape(output[2])[-1])))

  bbox = tf.concat(boxes, axis=1)
  confidence = tf.concat(conf, axis=1)
  class_probs = tf.concat(out_type, axis=1)

  scores = confidence * class_probs
  
  boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
    scores=tf.reshape(
        scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
    max_output_size_per_class=100,
    max_total_size=100,
    iou_threshold=yolo_iou_threshold,
    score_threshold=yolo_score_threshold
  )
  
  return boxes, scores, classes, valid_detections

def YoloConv(filters, name=None):
  def yolo_conv(x_in):
    if isinstance(x_in, tuple):
      inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
      x, x_skip = inputs

      x = DarknetConv(x, filters, 1)
      x = UpSampling2D(2)(x)
      x = Concatenate()([x, x_skip])
    else:
      x = inputs = Input(x_in.shape[1:])

    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, filters, 1)
    return Model(inputs, x, name=name)(x_in)
  return yolo_conv

def YoloOutput(filters, anchors, classes, name=None):
  def yolo_output(x_in):
    x = inputs = Input(x_in.shape[1:])
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
    x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                        anchors, classes + 5)))(x)
    return tf.keras.Model(inputs, x, name=name)(x_in)
  return yolo_output

#Crate custom layers and model -------- Darknet : --------------- #
def DarknetConv(x, filters, size, strides=1, batch_norm=True):
  if strides == 1:
    padding = 'same'
  else:
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
    padding = 'valid'
  x = Conv2D(filters=filters, kernel_size=size,
          strides=strides, padding=padding,
          use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
  if batch_norm:
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
  return x

def DarknetResidual(x, filters):
  previous  = x
  x = DarknetConv(x, filters // 2, 1)
  x = DarknetConv(x, filters, 3)
  x = Add()([previous , x])
  return x

def DarknetBlock(x, filters, blocks):
  x = DarknetConv(x, filters, 3, strides=2)
  for _ in repeat(None, blocks):
    x = DarknetResidual(x, filters)       
  return x

def Darknet(name=None):
  x = inputs = Input([None, None, 3])
  x = DarknetConv(x, 32, 3)
  x = DarknetBlock(x, 64, 1)
  x = DarknetBlock(x, 128, 2)
  x = x_36 = DarknetBlock(x, 256, 8)
  x = x_61 = DarknetBlock(x, 512, 8)
  x = DarknetBlock(x, 1024, 4)
  return tf.keras.Model(inputs, (x_36, x_61, x), name=name)
#--end Darknet Definee layers and model:-----end Darknet : ------- #

def draw_outputs(img, outputs, class_names):
  boxes, score, classes, nums = outputs
  boxes, score, classes, nums = boxes[0], score[0], classes[0], nums[0]
  wh = np.flip(img.shape[0:2])
  for i in range(nums):
    x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
    x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
    img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
    img = cv2.putText(img, '{} {:.4f}'.format(
      class_names[int(classes[i])], score[i]),
      x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
  return img
# -------------------------------- end: DEFINE ------------------------------------#
# load weight darkness ---------------------------------------
#
#
# end load daeekness -----------------------------------------

yolo = YoloV3(classes=num_classes)  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   load weight hể:
yolo.load_weights('checkpoints/yolov3.tf').expect_partial()

if __name__ == "__main__":
	initdetect()
	app.run()
