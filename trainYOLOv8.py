from ultralytics import YOLO

def train_model_Yolo(yolo_version, data_yaml):
    model = YOLO(yolo_version)
    model.train(data=data_yaml, epochs=30, imgsz=640, batch=16)
    
if __name__ == '__main__':
    
    data_yaml = 'C:/Users/Hooman/Desktop/ML_github/Projects/My Projects/Number-Plates-Recognition/Data/data.yaml'
    yolo_version = 'yolo11n.pt'
    # train_model_Yolo(yolo_version, data_yaml)