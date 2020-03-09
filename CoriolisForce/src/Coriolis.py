#!/usr/bin/env python
# coding:utf-8
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from math import pi, sin, cos, atan
from src.Loger import Loger

f_output = '../data/output.png'
v_output = "../data/output.mp4"
radius = 200
nx = ny = 800
x0 = y0 = int(nx/2)

def make_circle():
    data = np.zeros((nx, ny, 3), dtype=np.uint8)
    cv2.circle(data, (x0, y0), radius, (0, 0, 255))
    cv2.circle(data, (x0, y0), 3, (0, 0, 255), thickness=-1)
    return data

def plot_point(x, y, data, color):
    color = tuple(int(x) for x in color)
    cv2.circle(data, (int(round(x)), int(round(y))), 3, color, thickness=-1)
    return

def print_word(data, v):
    temp = np.zeros((nx, ny, 3), dtype=np.uint8)
    im = Image.fromarray(temp)
    draw = ImageDraw.Draw(im)
    font1 = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 20)
    font2 = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 30)
    draw.text((20, 30), "喷水速度/转速=", (0, 140, 0), font=font1)
    draw.text((170, 22), "%g" % v, (0, 255, 0), font=font2)
    data += np.array(im)

class Drip():
    def __init__(self, x, y, v, beta, color):
        self.xy = np.array((x, y))
        self.v = np.array((v*cos(beta), v*sin(beta)))
        self.color = np.array(color) * 1.
        self.decay = 0.99
        self.valid = True

    def move(self):
        self.xy += self.v
        self.color *= self.decay
        if self.xy.min() < 10 or self.xy.max() > nx:
            self.valid = False

def plot_one_movie(data_circle, videoWriter, drip_list, nv):
    frames, drip_step = 200, 2
    omega = 2 * pi / frames
    alpha = atan(nv)
    v = omega * radius * nv
    data_circle_word = data_circle.copy()
    print_word(data_circle_word, nv)
    for t in range(frames + 1):
        idata = data_circle_word.copy()
        theta = t * omega
        x1 = x0 + radius * cos(theta)
        y1 = y0 + radius * sin(theta)
        x2 = x0 + radius * cos(theta + pi)
        y2 = y0 + radius * sin(theta + pi)
        plot_point(x1, y1, idata, (0, 255, 255))                   # 画喷嘴1
        plot_point(x2, y2, idata, (0, 255, 255))                   # 画喷嘴2
        for drip in drip_list:
            if drip.valid:
                drip.move()
                plot_point(drip.xy[0], drip.xy[1], idata, drip.color)  # 画水滴
        if t % drip_step == 0:                                      # 喷水滴
            beta = alpha + theta + pi / 2
            new_drip1 = Drip(x1, y1, v, beta, (255, 255, 100))
            new_drip2 = Drip(x2, y2, v, beta+pi, (255, 100, 255))
            drip_list.append(new_drip1)
            drip_list.append(new_drip2)
        videoWriter.write(idata)
    loger.log("done speed=%s" % nv)

def plot_movie(data_circle):
    fps = 25
    drip_list = []
    videoWriter = cv2.VideoWriter(v_output, cv2.VideoWriter_fourcc(*'MP4V'), fps, (nx, ny))
    plot_one_movie(data_circle, videoWriter, drip_list, 0.2)
    plot_one_movie(data_circle, videoWriter, drip_list, 0.5)
    plot_one_movie(data_circle, videoWriter, drip_list, 0.8)
    plot_one_movie(data_circle, videoWriter, drip_list, 0.9)
    plot_one_movie(data_circle, videoWriter, drip_list, 1)
    plot_one_movie(data_circle, videoWriter, drip_list, 5)

loger = Loger()
loger.log('begin')
data_circle = make_circle()
cv2.imwrite(f_output, data_circle)
plot_movie(data_circle)
loger.logu('done.')

