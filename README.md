# Object-Measure

This project is a **computer vision-based measurement tool** that processes a video to detect, classify, and measure three types of objects:
- **Washers**
- **Screws**
- **Hex nuts**

Using a known reference object (a red object of known width), the script calculates the real-world dimensions of the objects in inches and reports measurement accuracy across the video frames.

## 📝 Features

- Detects reference objects based on red color masking.
- Identifies and classifies:
  - Washers (roughly circular with small area)
  - Screws (long, narrow shapes)
  - Hex nuts (small area, not overlapping other objects)
- Measures width and height of each object.
- Computes average dimensions for each object type.
- Calculates measurement errors compared to known dimensions.
- Supports interactive pausing (`p` key) and quitting (`q` key) while processing the video.

## 🔧 Requirements
    We recommend using Python 3.12 or later for this script.
    
    Create a virtual environment and install the required packages:
        
        python -m venv .venv
        .venv\Scripts\activate
        pip install -r requirements.txt
    
    Usage:
    
        python measure.py --video <path_to_video> --width <known_width>
        
    Example:
    
        python measure.py --video OneofEach.mp4 --width 1.0
