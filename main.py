from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate, write_csv

mot_tracker = Sort()

results = {}
car_model = YOLO(
    "C:/Users/Hooman/Desktop/ML_github/Projects/My Projects/Number-Plates-Recognition/Using YOLO And esayORC/models/yolo11n.pt"
)
license_plate_model = YOLO(
    "C:/Users/Hooman/Desktop/ML_github/Projects/My Projects/Number-Plates-Recognition/Using YOLO And esayORC/models/best.pt"
)

# Load Video
cap = cv2.VideoCapture(
    "C:/Users/Hooman/Desktop/ML_github/Projects/My Projects/Number-Plates-Recognition/Using YOLO And esayORC/videos/sample.mp4"
)

vehicles = [2, 3, 5, 7]

# Read Frames
frame_num = -1
ret = True
while ret:
    frame_num += 1
    ret, frame = cap.read()
    if ret :
        results[frame_num] = {}

        # Detect cars
        car_detection = car_model(frame)[0]
        detections_ = []
        for cars in car_detection.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = cars
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track Vehicles
        track_ids = mot_tracker.update(np.array(detections_))

        # Detect the License Plate
        license_plates_detection = license_plate_model(frame)[0]
        for license_plate in license_plates_detection.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign License Plate to Car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # Crop Licenseplates
            license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

            # Process licenseplates
            license_plate_crop_gray = cv2.cvtColor(
                license_plate_crop, cv2.COLOR_BGR2GRAY
            )
            _, license_plate_crop_gray_threshold = cv2.threshold(
                license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
            )

            # Read Licenseplates
            license_plate_text, license_plate_text_score = read_license_plate(
                license_plate_crop_gray
            )

            if license_plate_text is not None:
                results[frame_num][car_id] = {
                    "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                    "license_plate": {
                        "bbox": [x1, y1, x2, y2],
                        "text": license_plate_text,
                        "bbox_score": score,
                        "text_score": license_plate_text_score,
                    },
                }
# Write Licenseplates
write_csv(results,'./test.csv')