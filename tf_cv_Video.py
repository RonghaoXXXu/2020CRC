import cv2 as cv
import tensorflow as tf
import time

start_time = time.clock()


# Read the graph.
num_classes=['armor_blue','armor_red']
weight, hight=300,300

with tf.io.gfile.GFile('/home/ronghao/Desktop/Save_dir_RM/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

cap=cv.VideoCapture(0)
p=bool(1)
with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    while True:
        count = [([] * 2) for i in range(len(num_classes))]

        flag,img=cap.read()
        if flag is False:
            break

        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (weight, hight))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        thresold=0.1
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = round(float(out[1][0][i]),2)
            bbox = [float(v) for v in out[2][0][i]]
            if score > thresold:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows

                for i in range(len(num_classes)):
                    if i==(classId-1):
                        count[i].append(classId)
                
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                cv.putText(img, num_classes[classId-1]+':'+str(score), (int(x), int(y)),cv.FONT_HERSHEY_DUPLEX,0.6, (0,255,0), 1)
        cv.imshow('TensorFlow MobileNet-SSD', img)
        
        work=time.clock()-start_time
        if work>20 and p:
            print('START')
            for j in range(len(num_classes)):
                if len(count[j])>0:
                    result = "Goal_ID=%s  ; Num=%d" %(num_classes[j],len(count[j]))
                    print(result)
            print('END')
            with open('result.txt','wt') as file_handle:
                file_handle.write('START\n')
                for j in range(len(num_classes)):
                    if len(count[j])>0:
                        result = "Goal_ID=%s  ; Num=%d\n" %(num_classes[j],len(count[j]))
                        file_handle.write(result)
                file_handle.write('END\n')
            p=0
            
cap.release()
cv.destroyAllWindows()
