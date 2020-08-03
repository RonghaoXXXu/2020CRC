#Demo_WDJ
-这是一个识别武大靖的demo。主要理清识别思路。
##概述
-本项目使用tensorflow==2.X深度学习框架，借助Object_detection_API进行识别模型训练
-利用opencv—dnn读取模型
-使用openvino进行模型推理加速
-视频操控
-实时目标识别
##过程实现
###tensorflow==2.X--Object_detection_API环境安装
####tensorflow安装，protocf>3.3安装
####Object_detection_API环境安装
% /home/feyker
mkdir models
cd models
git clone ....../tensorflow/models
protoc object_detection/protos/*.proto --python_out=.
%vim ¬/.bashrc
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
`pwd`=..../models/research
% /home/feyker.../models
python object_detection/builders/model_builder_test.py

出现：run ...  OK...!
###模型构建
####数据准备
%/home/feyker.../models/research/object_detection
建议用绝对路径，便于移植。
mkdir   XXX-imgs----存放训练图片，xml，一定清晰可见，准确有效。暂时不用分train和eval集。
            data----存放cvs,record,label_map.pbtxt
        training----存放模型config，训练过程中分步保存的idex，meda，0000001，eval0
mkdir  XXX-model----存放转化过的可用模型

labelImg----标记后生成对应的xml,。

修改     xml_cvs.py ----1,image_path=='XXX-imgs'
                   ----2,cvs_input_path=='data'
                   ----3,row_labels,识别种类个数name,return id。跟label_map.pbtxt照应。
                   ----4,3：1分成train，eval
修改cvs_tfrecord.py ----1,image_path=='XXX-imgs'
                   ----2,record_input_path=='data'
                   ----3,train，eval.record
python3 xml_cvs.py cvs_tfrecord.py

新建label_map.pbtxt
示例：
item{
  id=1
  name='WDJ'
}
...与label标定的时候照应

配置XXX.config
num_classes=1----识别种类数目
batch_size=1 ----数目越大，效果越好。但训练时间越长
train.record
eval.record
label_map.pbtxt
注释掉  model.ckpt

####开始训练
