#!/usr/bin/env python
# coding:utf-8
##################################################
#  Author: cosmo
#  Create time: 2020.06.24 16:23:51
#  Update time: 2020.06.27 14:25:45
##################################################
import os
import sys
import cv2
import math
import numpy as np
# from PIL import Image, ImageDraw, ImageFont

dir_input  = sys.argv[2]   # 全量照片所在目录
f_base_pic = sys.argv[1]   # 亮度参考基准照片(在以上目录中）
v_output   = sys.argv[3]   # 输出MP4文件名
fps    = int(sys.argv[4])  # 输出视频帧率

video_width = 1200         # 视频分辨率


# 转成二值图，识别日面
def getMask(image):
    mimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold = mimage.max() / 4
    mask = np.zeros_like(mimage)
    mask[np.where(mimage > threshold)] = 1
    return mask

def getMean(image, mask):
    mimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean0 = (mimage * (1 - mask)).sum() / (1 - mask).sum()
    mean1 = (mimage * mask).sum() / mask.sum()
    # 分通道算均值
    # mean0 = (image * (1 - mask[..., None])).sum(axis=(0, 1)) / (1 - mask).sum()
    # mean1 = (image * mask[..., None]).sum(axis=(0, 1)) / mask.sum()
    return mean0, mean1

# 计算亮部重心
def calc_center(bin_image):
    nx, ny = bin_image.shape
    xf = bin_image.sum(axis=1)
    yf = bin_image.sum(axis=0)
    xcenter = int(np.arange(nx).dot(xf) / bin_image.sum())
    ycenter = int(np.arange(ny).dot(yf) / bin_image.sum())
    return xcenter, ycenter

# 从重心点出发按某角度射线，寻找与圆弧交点
def find_edge_point(image, bin_image, center, beta):
    width = 5
    buff = []
    for s in range(bin_image.size):
        x = center[0] + int(s * math.cos(beta))
        y = center[1] + int(s * math.sin(beta))
        if x < 0 or y < 0 or x >= bin_image.shape[0] or y >= bin_image.shape[1]:
            break
        buff.append((x, y, bin_image[x, y]))
    arr_line = np.array([x[2] for x in buff])
    arr_line_roll = np.concatenate((arr_line[:1], arr_line[:-1]))
    arr_line_diff = arr_line_roll - arr_line
    diff_idxs = np.where(arr_line_diff > 0)[0]
    for diff_idx in diff_idxs[::-1]:
        diff_cord = buff[diff_idx][:-1]
        if 0 == arr_line_diff[diff_idx - width: diff_idx].sum() == arr_line_diff[diff_idx + 1: diff_idx+width].sum():
            return diff_cord
    return 0, 0

def distance(cx, cy, ex, ey):
    return math.sqrt((cx - ex)**2 + (cy - ey)**2)

# 用于判断弧线弯曲方向的长度基准
def calc_middle_length(a, b, beta):
    h = b * math.sin(2 * beta)
    a2 = b * math.cos(2 * beta)
    a1 = a - a2
    theta = math.atan(h / a1)
    return a * math.sin(theta) / math.sin(beta + theta)

# 过滤掉月面边缘点(二阶微分小于0)
def filter_moon_edges(list_edge_points, cx, cy, beta):
    ret = []
    list_distance = [distance(cx, cy, ex, ey) for ex, ey in list_edge_points]
    lens = len(list_distance)
    for i in range(lens-1):
        if list_distance[i] < 100: continue
        middle_distance = calc_middle_length(list_distance[i-1], list_distance[i+1], beta)
        if list_distance[i] >= middle_distance:
            ret.append(list_edge_points[i])
    middle_distance = calc_middle_length(list_distance[-2], list_distance[0], beta)
    if list_distance[-1] >= middle_distance:
        ret.append(list_edge_points[-1])
    return ret

# 最小二乘法算日心坐标
def fit_sun_center(list_sun_edge_points):
    assert len(list_sun_edge_points) > 2
    sx, sy, sxx, syy, sxy, sxxx, sxxy, sxyy, syyy = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for x, y in list_sun_edge_points:
        sx += x
        sy += y
        sxx += x*x
        sxy += x*y
        syy += y*y
        sxxx += x*x*x
        sxxy += x*x*y
        sxyy += x*y*y
        syyy += y*y*y
    N = len(list_sun_edge_points)
    C = N * sxx - sx * sx
    D = N * sxy - sx * sy
    E = N * sxxx + N * sxyy - sx * (sxx + syy)
    G = N * syy - sy * sy
    H = N * sxxy + N * syyy - sy * (sxx + syy)
    a = (H * D - E * G) / (C*G - D*D)
    b = (H * C - E * D) / (D*D - G*C)
    c = - (sxx + syy + a * sx + b * sy) / N
    ox = int(-a / 2)
    oy = int(-b / 2)
    R = int(math.sqrt(a*a + b*b - 4*c) / 2)
    print("\t# FIT SUN: %s" % str((ox, oy, R)))
    return ox, oy, R

# 梯度下降法修正日心坐标
def fit_sgd(a, b, points, R, alpha, rounds):
    def grad(a, b, points):
        ga, gb = 0., 0.
        for px, py in points:
            d = distance(px, py, a, b)
            ga += (d*d - R*R) * (a - px)
            gb += (d*d - R*R) * (b - py)
        return 4*ga/len(points), 4*gb/len(points)
    def loss(a, b, points):
        return sum((distance(a, b, px, py)**2 - R**2)**2 for px, py in points) / len(points)
    for i in range(rounds):
        ga, gb = grad(a, b, points)
        if math.fabs(ga) + math.fabs(gb) < 1:
            break
        a -= alpha * ga
        b -= alpha * gb
    print("\t# SGD SUN: %s" % str((int(a), int(b), R)))
    return int(a), int(b)

def filter_moon_points_by_distance(list_edge_points, ox, oy):
    distances = np.array([distance(ox, oy, ex, ey) for ex, ey in list_edge_points])
    mean = distances.mean()
    std = distances.std()
    return [list_edge_points[i] for i in np.where(distances > mean-std)[0]]

# 亮度归一化
def regulateBrightness(image, base_mean_dark, base_mean_sun):
    mask = getMask(image)
    mean0, mean1 = getMean(image, mask)
    final_image = np.copy(image)
    final_image[mask == 1] = np.clip(image[mask == 1] * (base_mean_sun / mean1), 0, 255)
    ## 暗部不做提亮
    # final_image[mask == 0] = np.clip(image[mask == 0] * (base_mean_dark / mean0), 0, 255)
    print('\t# image mean sun: %.4f /%.4f' % (mean1, base_mean_sun/mean1))
    return final_image

Radius = 0

def manage_jpg(f_input, base_mean_dark, base_mean_sun, for_radius=False):
    image = cv2.imread(os.path.join(dir_input, f_input))
    bin_image = getMask(image)
    cx, cy = calc_center(bin_image)

    # 从重心撒点，求圆心1
    period = 20
    list_edge_points = []
    for i in range(period):
        ex, ey = find_edge_point(image, bin_image, (cx, cy), 2 * math.pi * i /period)
        if ex != 0:
            list_edge_points.append((ex, ey))
    list_sun_edge_points = filter_moon_edges(list_edge_points, cx, cy, 2*math.pi/period)
    print("\t# sun edge points: %s/%s" % (len(list_sun_edge_points), len(list_edge_points)))
    cx, cy, R = fit_sun_center(list_sun_edge_points)

    # 从圆心1撒点，求圆心2
    period = 60
    list_edge_points = []
    for i in range(period):
        ex, ey = find_edge_point(image, bin_image, (cx, cy), 2 * math.pi * i /period)
        if ex != 0:
            list_edge_points.append((ex, ey))
    list_sun_edge_points = filter_moon_edges(list_edge_points, cx, cy, 2*math.pi/period)
    print("\t# sun edge points: %s/%s" % (len(list_sun_edge_points), len(list_edge_points)))
    ox, oy, R = fit_sun_center(list_sun_edge_points)

    # 再次过滤误识别的月面点，重新求圆心3
    list_sun_edge_points = filter_moon_points_by_distance(list_sun_edge_points, ox, oy)
    print("\t# sun edge point3: %s/%s" % (len(list_sun_edge_points), len(list_edge_points)))
    ox, oy, R = fit_sun_center(list_sun_edge_points)
    if for_radius:
        global Radius
        Radius = R
        print('Radius = %s' % Radius)
        return

    # 从圆心3起始，固定半径修正圆心4
    ox, oy = fit_sgd(ox, oy, list_sun_edge_points, Radius, 0.0000002, 50)

    half = int(Radius * 1.14)
    clip_image = image[ox-half: ox+half, oy-half: oy+half]
    clip_image = cv2.resize(clip_image, (video_width, video_width))
    final_image = regulateBrightness(clip_image, base_mean_dark, base_mean_sun)
    # print_word(final_image, f_input)
    return final_image

# 加文字
"""
def print_word(data, src_name):
    temp = np.zeros(data.shape, dtype=np.uint8)
    im = Image.fromarray(temp)
    draw = ImageDraw.Draw(im)
    fonts = ImageFont.truetype("/System/Library/Fonts/SFNSTextCondensed-Semibold.otf", 27)
    draw.text((20, 30),       "Photo: %s" % src_name, (70, 100, 110), font=fonts)
    draw.text((20, 1200-110), "DSLR: Canon 600D",   (70, 100, 110), font=fonts)
    draw.text((20, 1200-70),  "Telescope: APM107apo", (70, 100, 110), font=fonts)
    data += np.array(im)
"""

def main():
    print('read base data ...')
    image = cv2.imread(f_base_pic)
    masker = getMask(image)
    # 算基准亮度
    base_mean_dark, base_mean_sun = getMean(image, masker)
    print("base mean dark: " + str(base_mean_dark))
    print("base mean sun : " + str(base_mean_sun))
    # 算太阳基准半径
    manage_jpg(f_base_pic, base_mean_dark, base_mean_sun, for_radius=True)
    list_jpg = sorted(list(filter(lambda x: x[-4:] == '.jpg', os.listdir(dir_input))))
    print("# total jpgs: %s" % len(list_jpg))
    videoWriter = cv2.VideoWriter(v_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1200, 1200))
    for i, f_jpg in enumerate(list_jpg):
        print('[%3d] load image: %s' % (i, f_jpg))
        image = manage_jpg(f_jpg, base_mean_dark, base_mean_sun)
        videoWriter.write(image)
    videoWriter.release()
    print("done.")

main()

