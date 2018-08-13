#!/usr/bin/env python
"""
Creates an HTTP server with basic auth and websocket communication. Applies a SSD mobilenet and sends the results to a browser.
"""
import argparse
import base64
import hashlib
import os
import time
import threading
import webbrowser
import threading
import urllib
import tarfile

from PIL import Image, ImageFont, ImageDraw

try:
    import cStringIO as io
except ImportError:
    import io

import tornado.web
import tornado.websocket
from tornado.ioloop import PeriodicCallback

import numpy as np
import tensorflow as tf

# Hashed password for comparison and a cookie for login cache
ROOT = os.path.normpath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "password.txt")) as in_file:
    PASSWORD = in_file.read().strip()
COOKIE_NAME = "RaspiSSD"


class IndexHandler(tornado.web.RequestHandler):

    def get(self):
        if args.require_login and not self.get_secure_cookie(COOKIE_NAME):
            self.redirect("/login")
        else:
            self.render("index.html", port=args.port)


class LoginHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("login.html")

    def post(self):
        password = self.get_argument("password", "")
        if hashlib.sha512(password).hexdigest() == PASSWORD:
            self.set_secure_cookie(COOKIE_NAME, str(time.time()))
            self.redirect("/")
        else:
            time.sleep(1)
            self.redirect(u"/login?error")


class ErrorHandler(tornado.web.RequestHandler):
    def get(self):
        self.send_error(status_code=403)


class WebSocket(tornado.websocket.WebSocketHandler):

    def on_message(self, message):
        """Evaluates the function pointed to by json-rpc."""

        # Start an infinite loop when this is called
        if message == "read_camera":
            if not args.require_login or self.get_secure_cookie(COOKIE_NAME):
                self.camera_loop = PeriodicCallback(self.loop, 100)
                self.camera_loop.start()
            else:
                print("Unauthenticated websocket request")

        # Extensibility for other methods
        else:
            print("Unsupported function: " + message)

    def loop(self):
        """Sends camera images in an infinite loop."""
        lock.acquire()
        try:
            if args.detection == 'Yes':
                if frame_jpeg is not None:
                    self.write_message(base64.b64encode(frame_jpeg))

            else:
               sio = io.BytesIO()
               camera.capture(sio, "jpeg", use_video_port=True)
               self.write_message(base64.b64encode(sio.getvalue()))
        except tornado.websocket.WebSocketClosedError:
            self.camera_loop.stop()
        lock.release()

def recordThread(camera):

    if args.record == 'Yes':
        while True:
            for filename in camera.record_sequence(
                    'clip%02d.h264' % i for i in range(10)):
                print('Recording to %s' % filename)
                camera.wait_recording(10)
    else:
        stream = picamera.PiCameraCircularIO(camera, seconds=10)
        camera.start_recording(stream, format='h264')
        while True:
            camera.wait_recording(1)

def downloadDNN(MODEL_NAME):
    # What model to download.
     
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    print("Download Model:", DOWNLOAD_BASE + MODEL_FILE)

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())



def detectSSDThread(camera,lock):
    global frame_jpeg

    import csv
    csvfile = csv.reader(open("labels.txt"))
    classNames = dict(csvfile)
    
    MODEL_NAME = 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03'

    PATH_TO_CKPT = os.getcwd() + '/' + MODEL_NAME +'/frozen_inference_graph.pb'

    if not os.path.exists(PATH_TO_CKPT):
        downloadDNN(MODEL_NAME)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    font_path = "/usr/share/fonts/truetype/"
    ttf = ImageFont.truetype (font_path+'freefont/FreeMono.ttf', 10)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            #PREPROCESSING
            image_placeholder = tf.placeholder(tf.string)
            decode_jpeg = tf.image.decode_jpeg(image_placeholder)
            resize_image = tf.image.resize_images(decode_jpeg, [300,300])
            expand_image = tf.expand_dims(resize_image, axis=0)

            #POSTPROCESSING
            image_to_jpeg_placeholder = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
            jpeg_image = tf.image.encode_jpeg(image_to_jpeg_placeholder)
            while True:
                sio = io.BytesIO()
                lock.acquire()
                camera.capture(sio, "jpeg", use_video_port=True)
                lock.release()

                (frame, image_np_expanded) = sess.run([decode_jpeg, expand_image],feed_dict={image_placeholder: sio.getvalue()})

                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                start = time.time()
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                print("DNN Time:",time.time() - start)

                h, w, _ = frame.shape
                frame_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame_pil)
                for n in range(len(scores[0])):
                    
                    if scores[0][n] > 0.50:
                        ymin = int(boxes[0][n][0] * h)
                        xmin = int(boxes[0][n][1] * w)
                        ymax = int(boxes[0][n][2] * h)
                        xmax = int(boxes[0][n][3] * w)
                        draw.rectangle([xmin, ymin, xmax, ymax], outline = "red")

                        if str(int(classes[0][n])) in classNames.keys():
                            label = classNames[str(int(classes[0][n]))]+ ": " + str( scores[0][n])
                            draw.text((xmin,ymin), label, font=ttf, fill=(255,255,255,128))
                            print(label)  # print class and confidence

                frame_np = np.asarray(frame_pil)
                lock.acquire()

                frame_jpeg = sess.run(jpeg_image, feed_dict={image_to_jpeg_placeholder:frame_np})
                lock.release()
                time.sleep(0)


parser = argparse.ArgumentParser(description="Starts a webserver that "
                                 "connects to a webcam.")
parser.add_argument("--port", type=int, default=8000, help="The port on which to serve the website.")
parser.add_argument("--record", type=str, default="Yes", help="Record video to sdcard")
parser.add_argument("--detection", type=str, default="Yes", help="Send detection or original image")
parser.add_argument("--resolution", type=str, default="low", help="The video resolution. Can be high, medium, or low.")
parser.add_argument("--require-login", action="store_true", help="Require a password to log into webserver.")
args = parser.parse_args()

print("detection:",args.detection)
print("port:",args.port)
print(args.resolution)

import picamera
camera = picamera.PiCamera()
camera.framerate = 5
lock = threading.Lock()
frame_jpeg = None
resolutions = {"high": (1280, 720), "medium": (640, 480), "low": (320, 240)}
if args.resolution in resolutions:
    camera.resolution = resolutions[args.resolution]
else:
    raise Exception("%s not in resolution options." % args.resolution)


handlers = [(r"/", IndexHandler), (r"/login", LoginHandler),
            (r"/websocket", WebSocket),
            (r"/static/password.txt", ErrorHandler),
            (r'/static/(.*)', tornado.web.StaticFileHandler, {'path': ROOT})]
application = tornado.web.Application(handlers, cookie_secret=PASSWORD)
application.listen(args.port)

webbrowser.open("http://localhost:%d/" % args.port, new=2)

record_thread = threading.Thread(target=recordThread, args=(camera,))
record_thread.start()

detect_thread = threading.Thread(target=detectSSDThread, args=(camera,lock))
detect_thread.start()


tornado.ioloop.IOLoop.instance().start()
