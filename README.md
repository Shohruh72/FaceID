### Face Identification inference code using ONNX Runtime

AdaFace: Quality Adaptive Margin for Face Recognition
![Vizualization](https://github.com/Shohruh72/FaceID/blob/main/demo/output_demo.gif)

### Installation

```
conda create -n ONNX python=3.8
conda activate ONNX
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install onnxruntime-gpu==1.14.0
pip install opencv-python==4.5.5.64
```

### WebCam Inference
```bash
$ python main.py
```

## Model ONNX WEIGHT

| Backbone | Weight                                                                             |
|:--------:|------------------------------------------------------------------------------------|
|   R50    | [link](https://github.com/Shohruh72/FaceID/releases/download/v.1.0.0/recognition_r50.onnx) | 


#### Note

* This repo supports only inference, see reference for more details


#### Reference

* https://github.com/mk-minchul/AdaFace
* https://github.com/jahongir7174/FaceID
* https://github.com/Shohruh72/SCRFD
