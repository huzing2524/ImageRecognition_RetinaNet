# Create your views here.
# python3 manage.py runserver 192.168.2.162:8000
import os
import sys
sys.path.insert(0, "../ImageAI/retinanet/examples/ResNet50RetinaNet")
import requests

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView


def get_image(image_id):
    """发送请求，获取图片内容"""
    url = "http://192.168.2.89:8080/bg/file?id=" + image_id
    response = requests.get(url)
    image = response.content  # bytes
    print(image[:20], type(image))
    if image:
        return image
    else:
        return None


class ImagesRecognition(APIView):
    """DSD V2.3.0 首页图片识别功能 /api/v2/recognition/{id}"""

    def get(self, request, image_id):
        """接收图片，返回识别结果"""
        if not image_id:
            return Response({"msg": "缺少图片id"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image = get_image(image_id)

            if image is not None:
                saved_path = os.path.join(os.getcwd(), 'ImageAI/retinanet/finished')
                with open(saved_path + "/{}.jpg".format(image_id), "wb") as f:
                    f.write(image)
            else:
                return Response({"msg": "图片获取失败"}, status=status.HTTP_204_NO_CONTENT)
        except Exception:
                return Response({"msg": "图片下载出现异常"}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        from ImageAI.retinanet.examples import ResNet50RetinaNet
        try:
            result = ResNet50RetinaNet.predict(image_id)

            if result:
                return Response(result, status=status.HTTP_200_OK)
            else:
                return Response({"msg": "图片识别失败"}, status=status.HTTP_204_NO_CONTENT)
        except Exception:
            # print("异常")
            return Response(status=status.HTTP_503_SERVICE_UNAVAILABLE)
        finally:
            remove_path = os.path.join(os.getcwd(), 'ImageAI/retinanet/finished')
            if os.path.exists(remove_path + "/{}.jpg".format(image_id)):
                os.remove(remove_path + "/{}.jpg".format(image_id))


class Test(APIView):
    def get(self, request):
        return Response("hello django!")
