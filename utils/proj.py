from ultralytics import YOLO
import cv2
if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    model = YOLO("runs/detect/train5/weights/best.pt")
    # Train the model
    #model.train(data="datasets/data.yaml", epochs=10, batch=16, imgsz=640,freeze=11, workers=15)

    
    results = model(source="VIDEO_DATASET_FOR_THERMAL/OUT.mp4",show=True,save=True)
    cv2.waitKey(0)