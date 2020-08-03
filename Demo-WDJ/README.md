# Demo_WDJ
-这是一个识别武大靖的demo。主要理清识别思路。
## 概述
	1，本项目使用tensorflow==2.X深度学习框架，借助Object_detection_API进行识别模型训练
	2，利用opencv—dnn读取模型
	3，使用openvino进行模型推理加速
	4，视频操控
	5，实时目标识别
## 模型构建
### tensorflow==2.X--Object_detection_API环境安装
	tensorflow安装，protocf>3.3安装
	Object_detection_API环境安装
	
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
#### 数据准备
	%/home/feyker.../models/research/object_detection
	建议用绝对路径，便于移植。
	mkdir   XXX-imgs----存放训练图片，xml，一定清晰可见，准确有效。暂时不用分train和eval集。
            	data----存放cvs,record,label_map.pbtxt
        	training----存放模型config，训练过程中分步保存的idex，meda，0000001，eval0
	mkdir   XXX-model----存放转化过的可用模型

	labelImg----标记后生成对应的xml,。

	修改xml_cvs.py 
	
	1,image_path=='XXX-imgs'
        2,cvs_input_path=='data'
        3,row_labels,识别种类个数name,return id。跟label_map.pbtxt照应。
        4,3：1分成train，eval
	
        修改cvs_tfrecord.py 
	
	1,image_path=='XXX-imgs'
        2,record_input_path=='data'
        3,train，eval.record
	
	python3 xml_cvs.py cvs_tfrecord.py
#### 工具文件准备
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

#### 开始训练
	%...../object_detection
	train.txt
	python3 model_main.py \
		pipeline_config_path=training/XXX.config\
        	model_dir=training \
        	num_train_steps=训练集训练步数 \
        	num_eval_steps=测试集(评估)测试步数\
        	alsologtostderr
		
训练过程中保存的数据都在trainin里

最好在GPU上训练，否则巨慢。
#### 模型转化
	%...../object_detection
	python3 export_inference_graph.py\
 		input_type image_tensor\
 		pipeline_config_path training/XXX.config\
		trained_checkpoint_prefix training/model.ckpt-数字最大，步数最多\
 		output_directory 模型保存的地方
	%模型保存的地方
	frozen_inference_graph.pb
	pipline.config

## 读取模型
### opencvDNN+openvino
	%模型保存的地方
	python3 tf_text_graph_ssd.py\
		frozen_inference_graph.pb\
		pipline.config\
		WDJ_v2.pbtxt
	调用的文件
	frozen_inference_graph.pb
	WDJ_v2.pbtxt

详见于cvdnn_vino.cpp

### openvino+NCS2

	
