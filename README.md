# License Plate Detection

This repository, **License Plate Detection**, is a project for detecting and extracting information from license plates using **YOLO** (You Only Look Once) for object detection and **easyOCR** for Optical Character Recognition (OCR).

---

## Features

- **License Plate Detection**: Utilizes YOLOv11 for precise and fast object detection.
- **Text Extraction**: Extracts license plate text using easyOCR for further processing.
- **Visualization Tools**: Provides scripts to visualize detections and outputs.
- **Data Handling**: Includes tools for working with datasets, missing data handling, and output management.
- **Custom Model Training**: Contains functionality to train YOLOv11 on custom datasets.

---

## Project Structure

- **`.git` and `.gitignore`**: Files for version control and ignored files setup.
- **`CSV/`**: Directory for storing or processing related CSV files.
- **`main.py`**: The main entry point for running the detection and OCR pipeline.
- **`missing_data.py`**: Script to handle missing data in the dataset.
- **`models/`**: Directory containing pre-trained YOLOv8 models and configurations.
- **`sort/`**: Implementation of tracking methods to track license plates across frames.
- **`trainYOLOv11.py`**: Script for training the YOLOv11 model on a custom dataset.
- **`util.py`**: Utility functions to support the main detection and processing pipeline.
- **`visualize_output.py`**: Script to visualize detection results and OCR outputs.

---

## Requirements

Before running the project, ensure you have the following installed:

- Python 3.8 or later
- PyTorch
- easyOCR
- YOLOv11 (Ultralytics)
- OpenCV
- NumPy
- Pandas

To install the required packages, run:
```bash
pip install -r requirements.txt
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/HoomKh/License-Plate-Detection-YOLOV11-esayORC.git
cd License-Plate-Detection-YOLOV11-esayORC
```

### 2. Download or Prepare Dataset (Video)

Ensure the dataset for training or testing is prepared and placed in the appropriate directory. The dataset should include:
- **Images**: Containing license plates.
- **Annotations**: Label files compatible with YOLO format.
- **data.yaml**

You can download the dataset for testing from [here](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4).

[Here](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/) you can download the video for free.

### 3. Train the Model (Optional)

To train the YOLOv11 model on your custom dataset:
```bash
python trainYOLOv11.py
```

### 4. Run the Detection

Run the main detection script to detect license plates and extract text:
```bash
python main.py
```

### 5. Visualize Results

To visualize detection outputs:
```bash
python visualize_output.py
```

---

## Customization

- Modify `trainYOLOv11.py` for custom training parameters.
- Update paths in `main.py` to point to your specific data directories or pre-trained models.
- Add or update utility functions in `util.py` as needed.

---

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with detailed descriptions of changes.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Author

**HoomKh**  
GitHub: [HoomKh](https://github.com/HoomKh)

---
