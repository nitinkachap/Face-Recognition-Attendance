## Requirements
- Python 3.10 or 3.11 recommended
- Webcam
- Linux / Windows / macOS
- `face_recognition` (uses dlib), OpenCV, NumPy

## Setup on Arch Linux

Install system packages:

sudo pacman -S --needed python python-pip tk cmake gcc make

#Create and activate a virtual environment:

cd face_attendance_project
python -m venv .venv
source .venv/bin/activate


Install Python dependencies:

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

## Run
source .venv/bin/activate
python app.py


#reset code if shows cv2 error

sudo pacman -Syu python-opencv

deactivate 2>/dev/null
mv .venv .venv-old
/usr/bin/python -m venv
--system--site--packeges .venv
source .venv/bin/activate

#reinstall

python -m pip install setuptools==80.9.0
python -m pip install face_recognition
pillow numpy
