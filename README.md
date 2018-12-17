GPU环境需要安装Nvidia显卡驱动CUDA，CUDNN。在知道设备编号后，运行时可以指定使用哪个显卡运行程序。

指定ip和端口号在run.sh里面加上，例如CUDA_VISIBLE_DEVICES="1(设备编号)" python3 manage.py runserver 127.0.0.1:8000

/DSD/ImageAI/retinanet 目录文件是采用RetinaNet模型的图片识别，其它文件为Django服务器提供接口服务。

运行启动文件是 manage.py，使用下面命令开启服务。

```bash
python3 mannage.py runserver
```

