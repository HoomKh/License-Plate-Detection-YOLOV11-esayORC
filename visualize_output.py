import cv2
import numpy as np
import pandas as pd
import ast


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # -- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # -- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


# Load the results CSV file
results = pd.read_csv('./test_interpolated.csv')

# Load video
video_path = 'C:/Users/Hooman/Desktop/ML_github/Projects/My Projects/Number-Plates-Recognition/Using YOLO And esayORC/videos/sample.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError("Error opening video file. Please check the video path.")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

license_plate = {}

# Process each unique car
for car_id in np.unique(results['car_id']):
    max_score_idx = results[results['car_id'] == car_id]['license_number_score'].idxmax()
    max_frame = results.loc[max_score_idx]['frame_nmr']
    max_bbox = results.loc[max_score_idx]['license_plate_bbox']

    cap.set(cv2.CAP_PROP_POS_FRAMES, max_frame)
    ret, frame = cap.read()

    if not ret or frame is None:
        print(f"Skipping car_id {car_id} due to frame read failure.")
        continue

    try:
        x1, y1, x2, y2 = ast.literal_eval(max_bbox.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        height, width, _ = frame.shape
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            print(f"Invalid bounding box for car_id {car_id}: {max_bbox}")
            continue

        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

        license_plate[car_id] = {
            'license_crop': license_crop,
            'license_plate_number': results.loc[max_score_idx]['license_number']
        }
    except Exception as e:
        print(f"Error processing car_id {car_id}: {e}")
        continue

# Reset video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_nmr = -1
ret = True

# Process video frames
while ret:
    ret, frame = cap.read()
    frame_nmr += 1

    if not ret or frame is None:
        break

    df_ = results[results['frame_nmr'] == frame_nmr]

    for _, row in df_.iterrows():
        try:
            # Draw car bounding box
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(row['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # Draw license plate bounding box
            x1, y1, x2, y2 = ast.literal_eval(row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # Overlay license plate crop and text
            car_id = row['car_id']
            if car_id in license_plate and license_plate[car_id]['license_crop'] is not None:
                license_crop = license_plate[car_id]['license_crop']
                H, W, _ = license_crop.shape

                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[car_id]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                cv2.putText(frame,
                            license_plate[car_id]['license_plate_number'],
                            (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)
        except Exception as e:
            print(f"Error drawing for car_id {row['car_id']}: {e}")
            continue

    out.write(frame)

out.release()
cap.release()
