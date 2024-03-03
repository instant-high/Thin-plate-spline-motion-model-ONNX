# Thin-plate-spline-motion-model - onnx
Thin plate spline motion model (TPSMM) converted to ONNX

run inference:

python demo.py --source source.png --driving driving.mp4 --output output.mp4

optional parameters: --mode standard (default relative) and --cpu to run on cpu (default cuda)


I converted the models using modified code from https://github.com/TalkUHulk/Image-Animation-Turbo-Boost

Original TPSMM repo : https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model


