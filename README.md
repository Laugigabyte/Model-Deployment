# TFRecords file

  + TensorFlow提供了TFRecord的格式来统一存储数据，TFRecord格式是一种将图像数据和各种标签放在一起的二进制文件，在tensorflow中快速的复制、移动、读取、存储。特点：

  + 文件格式：.tfrecord or .tfrecords

  + 写入文件内容：使用Example将数据封装成protobuffer协议格式

  + 体积小：消息大小只需要xml的1/10~1/3 
  
  + 解析速度快：解析速度比xml快20~100倍

  + 每个example对应一张图片，其中包括图片的各种信息

# Model-Deployment

Applying tensorflow_serving for the model deployment

# 基于TensorFlow Serving的模型部署与使用

## 整体流程与架构：
![tYJklV.png](https://s1.ax1x.com/2020/06/02/tYJklV.png)
* Procedure 
  * Input
  * web server(Flask) + Tensorflow Serving Client
  * Tensorflow Serving
  * Output

# Details of Preparation
  * Install Tensorflow Serving
  ```   
  docker pull tensorflow/serving
  ```
  * Saving models: SavedModelBuilder module of tensorflow
  ```
    import os
    import tensorflow as tf
    slim = tf.contrib.slim
    import sys
    sys.path.append("../")

    from nets.nets_model import ssd_vgg_300

    data_format = "NHWC"

    ckpt_filepath = "../ckpt/fine_tuning/model.ckpt-0"


    def main(_):

        # 1、定义好完整的模型图，去定义输入输出结果
        # 输入:SSD 模型要求的数据（不是预处理的输入）
        img_input = tf.placeholder(tf.float32, shape=(300, 300, 3))

        # [300,300,3]--->[1,300,300,3]
        img_4d = tf.expand_dims(img_input, 0)

        # 输出:SSD 模型的输出结果
        ssd_class = ssd_vgg_300.SSDNet

        ssd_params = ssd_class.default_params._replace(num_classes=9)

        ssd_net = ssd_class(ssd_params)

        # 得出模型输出
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            predictions, localisations, _, _ = ssd_net.net(img_4d, is_training=False)

        # 开启会话，加载最后保存的模型文件是的模型预测效果达到最好
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 创建saver
            saver = tf.train.Saver()

            # 加载模型
            saver.restore(sess, ckpt_filepath)

            # 2、导出模型过程
            # 路径+模型名字："./model/commodity/"
            export_path = os.path.join(
                tf.compat.as_bytes("./model/myts"),
                tf.compat.as_bytes(str(1)))

            print("正在导出模型到 {}".format(export_path))

            # 建立builder
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # 通过该函数建立签名映射（协议）
            # tf.saved_model.utils.build_tensor_info(img_input)：填入的参数必须是一个Tensor
            prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    # 给输入数据起一个别名,用在客户端读取的时候需要指定
                    "images": tf.saved_model.utils.build_tensor_info(img_input)
                },
                outputs={
                    'predict0': tf.saved_model.utils.build_tensor_info(predictions[0]),
                    'predict1': tf.saved_model.utils.build_tensor_info(predictions[1]),
                    'predict2': tf.saved_model.utils.build_tensor_info(predictions[2]),
                    'predict3': tf.saved_model.utils.build_tensor_info(predictions[3]),
                    'predict4': tf.saved_model.utils.build_tensor_info(predictions[4]),
                    'predict5': tf.saved_model.utils.build_tensor_info(predictions[5]),
                    'local0': tf.saved_model.utils.build_tensor_info(localisations[0]),
                    'local1': tf.saved_model.utils.build_tensor_info(localisations[1]),
                    'local2': tf.saved_model.utils.build_tensor_info(localisations[2]),
                    'local3': tf.saved_model.utils.build_tensor_info(localisations[3]),
                    'local4': tf.saved_model.utils.build_tensor_info(localisations[4]),
                    'local5': tf.saved_model.utils.build_tensor_info(localisations[5]),
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
            )

            # 建立元图格式，写入文件
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'detected_model':
                        prediction_signature,
                },
                main_op=tf.tables_initializer(),
                strip_default_attrs=True)

            # 保存
            builder.save()

            print("Serving模型结构导出结束")

  ```

那么首先要确定我们的模型的输入输出，我们不管训练还是测试过程中输入图片会经过处理，输出也会经过后续处理。但是导出的模型原则是，让我们的输入输出最直接就是算法的输入输出，关于处理的逻辑可以在具体场景的业务下处理。
        需要注意的是输入和输出中填入的是单个Tensor，不支持列表结构。
## Tensorflow Serving部分
* Docker开启Serving服务，运行服务器
   
   ```
   docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source=/home/ubuntu/detectedmodel/commodity,target=/models/commodity/ -e MODEL_NAME=commodity -t tensorflow/serving

   ``` 

   其中，8500和8501分别对应gRPC和REST两个不同的端口：
   ```
   Port 8500 exposed for gRPC
   Port 8501 exposed for the REST API
   ```

   source改成服务器上模型导出的路径，-p指定端口；这样在服务器就回开启了8500以及8501两个端口供调用。

## Web Server + Tensorflow Serving Client部分
模型的服务已经开启了，并且通过gRPC和REST两种接口方式。我么那选择gRPC+ protobufs接口，进行获取，主要考虑的是性能。

Web Server + TensorFlow Serving Client，提供给互联网用户使用，需要一个Web服务器来托管TensorFlow服务客户端，需要我们使用tensorflow_serving.apis编写一个获取结果的客户端，也还是属于TensorFlow的程序。这样的话我们需要将Web环境与TensorFlow Serving相关环境放在一起。

编写requirements.txt:

```
grpcio==1.12.0
matplotlib==2.2.2
numpy==1.14.2
pandas==0.20.3
Pillow==4.3.0
tensorflow==1.8.0
flask==1.0.2
gunicorn==19.7.1
tensorflow-serving-api==1.10.1
```
btw :安装tensorflow 1.8.0的话图像处理的函数会出现版本问题，最后升到1.9解决。

编写Dockerfile:

```
FROM python:3.6-stretch
WORKDIR /tmp
COPY ./requirements.txt .
RUN pip install -i https://pypi.douban.com/simple -r requirements.txt
```

进入Dockerfile所在目录：

![tYItHO.png](https://s1.ax1x.com/2020/06/02/tYItHO.png)

使用docker命令建立容器，输入：

```
docker build -t tf-web .
```

[![tYIvG9.md.png](https://s1.ax1x.com/2020/06/02/tYIvG9.md.png)](https://imgchr.com/i/tYIvG9)

建立后查看容器：

![tYo0dU.png](https://s1.ax1x.com/2020/06/02/tYo0dU.png)

Web Server + Tensorflow Serving Client部分的代码均在web_code文件夹中

![tYoHSA.png](https://s1.ax1x.com/2020/06/02/tYoHSA.png)

其中web_code还包含了输入（此处为图像）预处理代码（image_preprocessing.py）及后处理代码（image_tag_result.py)；prediction.py代码用于向Tensorflow服务器发送请求并预处理响应。

向Server发送请求获取结果,运行结果获取值
想要去获取模型服务数据，需要使用到grpc与tensorflow_serving.apis相关库。API如下：

grpc:用来建立连接，返回一个grpc.Channel

grpc.insecure_channel('192.168.9.166:8500'):指定建立连接的IP+端口,本地测试则使用本地IP
tensorflow_serving.apis

prediction_service_pb2_grpc:
    stub = PredictionServiceStub(channel)打开通道，提供了可以向模型服务器提供请求的渠道

predict = stub.Predict(request)：发送数据，获取结果

result.outputs[]:通过指定模型服务中outputs名称来获取数据，返回TensorInfo.

predict_pb2：predict_pb2.PredictRequest()用于创建预测结果请求，返回request

request需要进行设置信息

request.model_spec.name：设置需要请求模型的名称

request.model_spec.signature_name：设置签名名称，如导出模型设置的"detected_model"

request.inputs['images']:获取处理后的数据提供给模型服务中的inputs "images"名称

```
# 打开与Tensorflow Serving服务连接
with grpc.insecure_channel('192.168.9.166:8500') as channel:
    # 打开通道，提供了可以向模型服务器提供请求的渠道
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # 创建请求
    request = predict_pb2.PredictRequest()
    # 请求的模型名
    request.model_spec.name = 'commodity'
    # 请求的模型签名
    request.model_spec.signature_name = 'detected_images'
    # 请求的输入
    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(img_tensor, shape=[300, 300, 3]))

    # 获取结果
    result = stub.Predict(request)

    # 解析结果
    predictions = [
        tf.convert_to_tensor(result.outputs['predict0']),
        tf.convert_to_tensor(result.outputs['predict1']),
        tf.convert_to_tensor(result.outputs['predict2']),
        tf.convert_to_tensor(result.outputs['predict3']),
        tf.convert_to_tensor(result.outputs['predict4']),
        tf.convert_to_tensor(result.outputs['predict5']),
    ]
    localisations = [
        tf.convert_to_tensor(result.outputs['local0']),
        tf.convert_to_tensor(result.outputs['local1']),
        tf.convert_to_tensor(result.outputs['local2']),
        tf.convert_to_tensor(result.outputs['local3']),
        tf.convert_to_tensor(result.outputs['local4']),
        tf.convert_to_tensor(result.outputs['local5']),
    ]

    # 运行Tensor获取值
    with tf.Session() as sess:
        p, l = sess.run([predictions, localisations])
```

# Deployment of Server
为了能够上线使用，我们会在服务器中进行部署。有两个需要进行服务运行的。一个是TensorFlow Serving，另一个是Flask Web程序。服务器上必须先使用Docker安装过TensorFlow Serving容器和Web环境容器的。

## 开启Tensorflow Serving
上传导出的模型及相关本地文件至服务器。

![ttmrz4.png](https://s1.ax1x.com/2020/06/02/ttmrz4.png)

以上为所需上传的文件。

开启tensorflow serving:

```
docker run -p 8501:8501 -p 8500:8500 --mount type=bind,source=/home/ubuntu/detectedmodel/commodity,target=/models/commodity -e MODEL_NAME=commodity -t tensorflow/serving
```

开启此前制作好的镜像：
```
docker run -t -p 80:5000 -v /home/ubuntu/web_code:/app tf-web
```
开启Web服务之后，我们就可以通过指定Web的接口使用前端平台访问。

[![ttulvQ.png](https://s1.ax1x.com/2020/06/02/ttulvQ.png)](https://imgchr.com/i/ttulvQ)

