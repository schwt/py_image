# solar_eclipse_alignment

用于日食照片的对齐生成视频.
主要用几何方法计算太阳边缘，过滤月球边缘，找到太阳圆心，对齐每张照片后合成视频.

#### requirements
- 电脑安装有python环境，已经cv2, numpy 两个python包
- 日食照片需提前从raw解出jpg。程序只处理jpg格式，对曝光度只能做微调。曝光、色温需提前处理好，筛检删掉曝光差异过大、越出视野等异常照片。
- 从照片中选择一张日面尽量大、曝光量最合适、清晰的照片作为基准

#### usage 
```
./solar_eclipse_alignment.py ./jpgs/IMG_2000.jpg ./jpgs/ output.mp4 10
```
如上示例中:
	照片所在目录为 ./jpgs/，基准照片为 IMG_2000.jpg， 输出视频名为 output.mp4，视频帧率为10
