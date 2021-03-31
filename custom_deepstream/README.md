# Custom_deepstream

## Apply YOLOV4 Deepstream
현재 Repo에 https://drive.google.com/file/d/1lx3-k_R88zBGivqEKJwf28x4Hiw3sWQ9/view?usp=sharing 에서 YOLOV4 Engine을 다운받아 넣은 뒤 다음 명령어를 사용하여 실행시킨다. 
(환경에 따라 경로가 다를 수 있으니 Rebuild 하여 사용한다.)
```bash
# Rebuild Deepstream
sudo make clean
sudo make

# Run Deepstream
./deepstream-app -c ./deepstream_app_config_yoloV4.txt

```

기존의 YOLOV3로 구현되어있던 Deepstream을 YOLOV4에 맞게 동작시키기 위해 다음과 같은 과정을 거쳤다. (Deepstream 5.0 기준)
YOLOV4 weight는 아래 링크에서 받을 수 있다.
https://github.com/AlexeyAB/darknet#pre-trained-models

## DarkNET(Pytorch) to ONNX
```bash
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
cd pytorch-YOLOv4
pip install onnxruntime
python demo_darknet2onnx.py yolov4.cfg yolov4.weights ./data/giraffe.jpg 1
```

## ONNX to TensorRT
```bash
trtexec --onnx=<onnx_file> --explicitBatch --saveEngine=<tensorRT_engine_file> --workspace=<size_in_megabytes> --fp16
# example
# trtexec --onnx=yolov4_1_3_608_608.onnx --explicitBatch --saveEngine=yolov4_1_3_608_608_fp16.engine --workspace=4096 --fp16
```

## YOLOV4 Custom implementation
nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp 에 아래의 항목을 추가한다.
```cpp
extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static NvDsInferParseObjectInfo convertBBoxYoloV4(const float& bx1, const float& by1, const float& bx2,
                                     const float& by2, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution

    float x1 = bx1 * netW;
    float y1 = by1 * netH;
    float x2 = bx2 * netW;
    float y2 = by2 * netH;

    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);
    x2 = clamp(x2, 0, netW);
    y2 = clamp(y2, 0, netH);

    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH);

    return b;
}

static void addBBoxProposalYoloV4(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxYoloV4(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV4Tensor(
    const float* boxes, const float* scores,
    const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    const uint& netW, const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;

    uint bbox_location = 0;
    uint score_location = 0;
    for (uint b = 0; b < num_bboxes; ++b)
    {
        float bx1 = boxes[bbox_location];
        float by1 = boxes[bbox_location + 1];
        float bx2 = boxes[bbox_location + 2];
        float by2 = boxes[bbox_location + 3];

        float maxProb = 0.0f;
        int maxIndex = -1;

        for (uint c = 0; c < detectionParams.numClassesConfigured; ++c)
        {
            float prob = scores[score_location + c];
            if (prob > maxProb)
            {
                maxProb = prob;
                maxIndex = c;
            }
        }

        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
        {
            addBBoxProposalYoloV4(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
        }

        bbox_location += 4;
        score_location += detectionParams.numClassesConfigured;
    }

    return binfo;
}

static bool NvDsInferParseYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<NvDsInferParseObjectInfo> objects;

    const NvDsInferLayerInfo &boxes = outputLayersInfo[0]; // num_boxes x 4
    const NvDsInferLayerInfo &scores = outputLayersInfo[1]; // num_boxes x num_classes

    // 3 dimensional: [num_boxes, 1, 4]
    assert(boxes.inferDims.numDims == 3);
    // 2 dimensional: [num_boxes, num_classes]
    assert(scores.inferDims.numDims == 2);

    // The second dimension should be num_classes
    assert(detectionParams.numClassesConfigured == scores.inferDims.d[1]);
    
    uint num_bboxes = boxes.inferDims.d[0];

    // std::cout << "Network Info: " << networkInfo.height << "  " << networkInfo.width << std::endl;

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeYoloV4Tensor(
            (const float*)(boxes.buffer), (const float*)(scores.buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV4 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4);
```

## Build Deepstream
아래의 명령어를 입력하여 build한다.
```bash
sudo make
```

## UDP Flag
deepstream_app.c 에 아래 항목을 추가하여 Local(127.0.0.1) 4729 port로 detection flag packet을 보내도록 하였다. 비행기나 새가 오른쪽 상단과 왼쪽 상단을 제외한 위치에 detect 되었을 때 detection flag packet 이 보내진다.
```cpp
if (((obj->rect_params.left<200|obj->rect_params.left+ obj->rect_params.width>1700)&obj->rect_params.top<400)|(strcmp(obj->obj_label, "bird") && strcmp(obj->obj_label, "aeroplane")))  // non detection for both sides and detect only bird and aeroplane
      {
        appCtx->show_bbox_text=0;
      } else {
        appCtx->show_bbox_text=1;
      }

        if (!appCtx->show_bbox_text)
        {
            obj->rect_params.border_width = 0;
        }

      if (!appCtx->show_bbox_text)
        continue;
      else
      {
          sockfd = socket(AF_INET, SOCK_DGRAM, 0);
          memset(&servaddr, 0, sizeof(servaddr));
          servaddr.sin_family = AF_INET;
          servaddr.sin_port = htons(PORT);
          servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
          sendto(sockfd, (const char *)det, strlen(det),
          MSG_CONFIRM, (const struct sockaddr *) &servaddr,
          sizeof(servaddr));
          close(sockfd);
      }
```
