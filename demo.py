from tqdm import tqdm
#import time
import numpy as np
import cv2
from skimage import img_as_ubyte
import multiprocessing
import onnxruntime
import argparse

from scipy.spatial import ConvexHull

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime demo")
    parser.add_argument("--source", type=str, help="input source image")
    parser.add_argument("--driving", type=str, help="input driving video")
    parser.add_argument("--output", default="./generated_onnx.mp4", type=str, help="generated video path")
    parser.add_argument("--mode", default='relative', choices=['standard', 'relative'])
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    return parser

def relative_kp(kp_source, kp_driving, kp_driving_initial):
    # Accessing the first element of the 3D arrays
    source_area = ConvexHull(kp_source[0]).volume
    driving_area = ConvexHull(kp_driving_initial[0]).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_value_diff = (kp_driving - kp_driving_initial)
    kp_value_diff *= adapt_movement_scale
    kp_new = kp_value_diff + kp_source

    return kp_new
    

def main():
    args = make_parser().parse_args()  
    
    if args.cpu:
        device = 'cpu'
    else:
        device = 'cuda'   
    
    session_options = onnxruntime.SessionOptions()
    #session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = multiprocessing.cpu_count()
    providers = ["CPUExecutionProvider"]
    if device == 'cuda':
      providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
      
    kp_detector = onnxruntime.InferenceSession('checkpoints/kp_detector.onnx', sess_options=session_options, providers=providers)    
    tpsm_model = onnxruntime.InferenceSession('checkpoints/tpsmm_rel.onnx', sess_options=session_options, providers=providers)    


    cap = cv2.VideoCapture(args.driving)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outvideo = cv2.VideoWriter(args.output, fourcc, 25, (256 , 256), True)
    
    # source image
    source = cv2.imread(args.source)
    source = cv2.resize(source, (256, 256))
    cv2_source = source.astype('float32') / 255
    source = cv2.cvtColor(cv2_source, cv2.COLOR_BGR2RGB)
    source = np.transpose(source[np.newaxis].astype(np.float32), (0, 3, 1, 2))

    ort_inputs = {kp_detector.get_inputs()[0].name: source}
    kp_source = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]  # 1, 50, 2

    # first frame
    ret, frame = cap.read()
    frame_face = cv2.resize(frame, (256, 256))/ 255
    frame_face = np.transpose(frame_face[np.newaxis].astype(np.float32), (0, 3, 1, 2))
    ort_inputs = {kp_detector.get_inputs()[0].name: frame_face}
    kp_driving_initial = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        frame_face = cv2.resize(frame, (256, 256))/ 255
        frame_face = np.transpose(frame_face[np.newaxis].astype(np.float32), (0, 3, 1, 2))

        driving = frame_face
        
        ort_inputs = {kp_detector.get_inputs()[0].name: driving}
        kp_driving = kp_detector.run([kp_detector.get_outputs()[0].name], ort_inputs)[0]
        
        if args.mode == 'standard':
            kp_norm = kp_driving
        elif args.mode=='relative':
            kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,kp_driving_initial=kp_driving_initial)
                                    
        ort_inputs = {tpsm_model.get_inputs()[0].name: kp_source,tpsm_model.get_inputs()[1].name: source, tpsm_model.get_inputs()[2].name: kp_norm, tpsm_model.get_inputs()[3].name: driving}
        out = tpsm_model.run([tpsm_model.get_outputs()[0].name], ort_inputs)[0]

        im = np.transpose(out.squeeze(), (1, 2, 0))
        #im = im.astype(np.float32)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Out",im)
        cv2.waitKey(1)

        outvideo.write(img_as_ubyte(im))
        
    cap.release()
    outvideo.release()


if __name__ == "__main__":
    main()
