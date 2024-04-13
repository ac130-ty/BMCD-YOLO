import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/BMCD-YOLO/ultralytics/cfg/models/v8/yolov8-C2f_Pmb-dyhead.yaml')
    #model.load('yolov8n.pt')
    model.train(data='D:/BMCD-YOLO/dataset/guSui/guSui.yaml',
                epochs=1000,
                batch=18,
                #resume='D:/BMCD-YOLO/runs/train/train33/weights/last.pt',
                project='runs/train',
                patience=50,
                lr0=0.1,
                lrf=0.1,
                warmup_bias_lr=0.0
    )