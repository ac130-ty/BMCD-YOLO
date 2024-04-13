import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('D:/BMCD-YOLO/runs/train/yolov8-C2f_Pmb-dyhead/weights/best.pt')
    
    model.val(#data='D:/BMCD-YOLO/dataset/guSui/guSui.yaml',
              data='D:/BMCD-YOLO/dataset/guSui/guSui.yaml',
              split='test',
              batch=18,
              project='runs/test',
              )