from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11s.pt")

    model.train(
        data="roboflow_dataset/data.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        workers=4,
        device=0,
        patience=30
    )

    # Predict on test images
    model.predict(
        source="roboflow_dataset/test/images",
        save=True,
        conf=0.5
    )