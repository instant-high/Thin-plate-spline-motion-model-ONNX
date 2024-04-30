# Thin-plate-spline-motion-model - onnx
Thin plate spline motion model (TPSMM) converted to ONNX

Update 2024-05-01

Faster inference code when running on GPU

Bug fix onnx conversion. removed bg_predictor.

driving - old version - new version:

https://github.com/instant-high/Thin-plate-spline-motion-model-ONNX/assets/77229558/51ac3ac0-a195-492e-ba54-1d6e3ce3f387

.

run inference:

python demo.py --source source.png --driving driving.mp4 --output output.mp4

optional parameters: --mode standard (default relative) and --cpu to run on cpu (default cuda)


I converted the models using modified code from https://github.com/TalkUHulk/Image-Animation-Turbo-Boost

Original TPSMM repo : https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model


