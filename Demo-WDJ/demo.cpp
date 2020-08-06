#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <map>
#include <string>
#include <time.h>

using namespace std;
using namespace cv;

//const size_t inWidth = 300;
//const size_t inHeight = 300;
//const float WHRatio = inWidth / (float)inHeight;

const char* classNames[]= {"background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
"fire hydrant", "background", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "background", "backpack",
"umbrella", "background", "background", "handbag", "tie", "suitcase", "frisbee","skis", "snowboard", "sports ball", "kite", "baseball bat","baseball glove", "skateboard", "surfboard", "tennis racket",
"bottle", "background", "wine glass", "cup", "fork", "knife", "spoon","bowl", "banana",  "apple", "sandwich", "orange","broccoli", "carrot", "hot dog",  "pizza", "donut",
"cake", "chair", "couch", "potted plant", "bed", "background", "dining table", "background", "background", "toilet", "background","tv", "laptop", "mouse", "remote", "keyboard",
"cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "background","book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"};

int main()
{
    clock_t start,finish;
    double totaltime;

    Mat img=imread("/home/youngho/Desktop/pictures/5.jpg");

    String weights = "/home/youngho/Desktop/demo/frozen_inference_graph.pb";
    String prototxt = "/home/youngho/Desktop/demo/graph.pbtxt";
    dnn::Net net=dnn::readNetFromTensorflow(weights,prototxt);

    start = clock();

    Mat blob=dnn::blobFromImage(img,1,Size(300,300));
    net.setInput(blob);
    Mat output=net.forward();

    Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    float thresold=0.0;
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence>thresold)
        {
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

            ostringstream ss;
            ss << confidence;
            String conf(ss.str());

            Rect object((int)xLeftBottom, (int)yLeftBottom,(int)(xRightTop - xLeftBottom),(int)(yRightTop - yLeftBottom));

            rectangle(img, object, Scalar(0, 0, 255), 2);
            cout << "objectClass:" << objectClass << endl;

            String label = String(classNames[objectClass]) + ": " + conf;
            cout << "label"<<label << endl;
            int baseLine = 0;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            rectangle(img, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),Size(labelSize.width, labelSize.height + baseLine)),
                          Scalar(0, 0, 255), FILLED);
            cv::putText(img, label, Point(xLeftBottom, yLeftBottom),FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 0));
        }
    }
    finish = clock();
    totaltime = finish - start;
    cout << "识别该帧图像所用的时间为：" << totaltime <<"ms"<< endl;

    namedWindow("result", 0);
    imshow("result", img);
    waitKey(0);
    return 0;
}
