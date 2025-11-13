# Solar Panel Defect Detection System

## What is this?

This is a computer program that automatically finds problems (defects) in solar panels by looking at thermal (heat) images. Think of it like a smart camera that can spot issues that might be hard for humans to see.

**What it does:**
- Analyzes thermal images of solar panels
- Detects 8 different types of defects (hotspots, bypassed diodes, open circuits, etc.)
- Can work in three ways:
  1. **Command Line** - Process many images at once from your computer
  2. **Web Interface** - Upload images through a web browser
  3. **Real-time Camera** - Connect a thermal camera for live detection

**What you need:**
- A Windows computer
- Python 3.10 or newer (free software)
- This project folder
- The dataset (training images) - see setup instructions below

---

## Quick Start Guide

**Just want to run it?** Follow these steps in order:

1. **Get the project files** (see "Getting Started" section below)
2. **Install Python** (if you don't have it)
3. **Set up the project** (one-time setup)
4. **Run the program** (see "How to Use" section)

---

## Getting Started

### Step 1: Get the Project Files

You have two options to get this project:

#### Option 1: Using Git (Recommended if you know Git)

**What is Git?** Git is a tool for downloading and managing code projects.

1. **Install Git** (if you don't have it):
   - Download from: https://git-scm.com/downloads
   - Run the installer and follow the instructions
   - Restart your computer after installing

2. **Download the project:**
   ```powershell
   git clone <repository-url>
   cd solar-defect-detection
   ```
   *(Replace `<repository-url>` with the actual web address of this project)*

3. **Get the large image files:**
   ```powershell
   git lfs install
   git lfs pull
   ```
   *(This downloads the training images which are stored separately)*

#### Option 2: ZIP File (Easier - No Git Required!)

**If someone sent you this project as a ZIP file:**

1. **Unzip the project** to a folder (e.g., `D:\solar-defect-detection`)

2. **Get the dataset folder separately:**
   - The ZIP file won't include the training images (they're too large)
   - Ask the person who sent you the project to share the `dataset/` folder separately
   - They can send it via: USB drive, file sharing service (Google Drive, Dropbox, etc.), or email (if small enough)

3. **Place the dataset folder:**
   - Put the complete `dataset/` folder inside the project folder
   - Make sure it looks like this:
     ```
     solar-defect-detection/
       dataset/
         train/
           images/   (contains .jpg image files)
           labels/   (contains .txt label files)
         valid/
           images/
           labels/
         test/
           images/
           labels/
       (other project files...)
     ```

4. **You're done with this step!** No Git needed - just continue to Step 2 below.

---

### Step 2: Install Python

**What is Python?** Python is a programming language. This project needs Python to run.

1. **Check if you have Python:**
   - Open PowerShell (search "PowerShell" in Windows Start menu)
   - Type: `python --version`
   - If you see a version number (like "Python 3.10.5"), you're good!
   - If you see an error, you need to install Python

2. **Install Python (if needed):**
   - Download from: https://www.python.org/downloads/
   - **Important:** During installation, check the box that says "Add Python to PATH"
   - Run the installer
   - Restart your computer after installing

3. **Verify installation:**
   - Open PowerShell again
   - Type: `python --version`
   - You should see a version number

---

### Step 3: Set Up the Project

**Open PowerShell in the project folder:**
1. Navigate to the project folder (e.g., `D:\solar-defect-detection`)
2. Right-click in the folder
3. Select "Open in Terminal" or "Open PowerShell window here"
   - OR open PowerShell and type: `cd D:\solar-defect-detection` (use your actual folder path)

**Create a virtual environment:**
*(This creates an isolated space for this project's software, so it doesn't interfere with other programs)*

```powershell
python -m venv .venv
```

**Activate the virtual environment:**
*(This "turns on" the isolated space)*

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` appear at the start of your command line - that means it's working!

**Install required software:**
*(This downloads and installs all the tools this project needs)*

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

This may take several minutes. Wait for it to finish.

---

### Step 4: Verify the Dataset

**Check that your dataset folder is set up correctly:**

1. Make sure you have a `dataset/` folder in the project
2. It should contain:
   - `train/images/` - Training images
   - `train/labels/` - Training labels (tells the program what defects are in each image)
   - `valid/images/` - Validation images
   - `valid/labels/` - Validation labels
   - `test/images/` - Test images
   - `test/labels/` - Test labels

3. **Update the paths** (if needed):
   - Open `dataset/data.yaml` in a text editor (like Notepad)
   - Change the paths to match where your project is located:
     ```yaml
     train: D:/solar-defect-detection/dataset/train/images
     val:   D:/solar-defect-detection/dataset/valid/images
     test:  D:/solar-defect-detection/dataset/test/images
     ```
   - Replace `D:/solar-defect-detection` with your actual folder path
   - Save the file

---

## How to Use

### Method 1: Quick Test (Easiest)

**Run detection on sample images:**

```powershell
python main.py
```

**What this does:**
- Automatically finds the best trained model (or uses a default one)
- Processes all images in the `sample/` folder
- Saves results with boxes drawn around detected defects
- Results appear in the `outputs/` folder

---

### Method 2: Train Your Own Model

**Train the AI to recognize defects:**

```powershell
python train_yolo.py
```

**What this does:**
- Teaches the computer to recognize defects using your training images
- Takes a long time (could be hours depending on your computer)
- Creates a trained model file at: `runs/detect/solar_defect_train/weights/best.pt`
- Shows progress and results as it trains

**To train for fewer rounds (faster, for testing):**
```powershell
$env:YOLO_EPOCHS="10"
python train_yolo.py
```
*(This trains for only 10 rounds instead of 50 - faster but less accurate)*

---

### Method 3: Predict on Validation Images

**Test the trained model on validation images:**

```powershell
python predict.py
```

**What this does:**
- Uses your trained model to find defects in validation images
- Saves annotated images showing what was detected
- Results appear in `runs/detect/predict*/` folder

---

### Method 4: Process Your Own Images

**Analyze images from any folder:**

```powershell
python -m software.run
```

**What this does:**
- Processes all images in the `sample/` folder by default
- Saves results to the `outputs/` folder

**To use your own images:**
```powershell
python -m software.run --input "D:\MyImages" --output "D:\Results"
```
*(Replace the paths with your actual folder locations)*

---

### Method 5: Web Interface (User-Friendly!)

**Launch a web browser interface:**

1. **Install web dependencies** (one-time):
   ```powershell
   pip install -r requirements-web.txt
   ```

2. **Start the web server:**
   ```powershell
   python -m webapp.run_web
   ```

3. **Open your web browser:**
   - Go to: `http://127.0.0.1:8000`
   - You'll see a page where you can upload images
   - Click "Choose File", select an image, then click "Analyze"
   - Results will appear on the right side

**What this does:**
- Creates a simple website on your computer
- Lets you upload images through your browser
- Shows results with detected defects highlighted
- No command line needed after starting!

---

### Method 6: Real-Time Camera Detection

**Use with a thermal camera (MLX90640):**

See `hardware/README_hardware.md` for detailed instructions.

```powershell
python hardware/real_time_detection.py
```

**What this does:**
- Connects to a thermal camera
- Shows live video with defect detection
- Requires special hardware (MLX90640 thermal camera)

---

## Where to Find Results

After running the program, your results will be in different places depending on what you ran:

- **Training results:** `runs/detect/solar_defect_train/`
  - Trained model: `weights/best.pt`
  - Training charts: `results.png`

- **Prediction results:** 
  - `runs/detect/predict*/` (from `predict.py`)
  - `outputs/` (from `main.py` or `software.run`)

- **Web app results:** `webapp/static/results/`

**To clean up old results and start fresh:**
```powershell
Remove-Item -Recurse -Force runs, outputs -ErrorAction SilentlyContinue
```

---

## Common Problems and Solutions

### Problem: "git is not recognized"

**Solution:**
- If you got this project as a ZIP file, you don't need Git! Use **Option 2** in the setup section
- If you want to use Git, install it from https://git-scm.com/downloads first

---

### Problem: "Git LFS pointer detected" or images are missing

**What this means:** The image files didn't download properly.

**Solution:**
- **If using Git (Option 1):** Run `git lfs install` then `git lfs pull`
- **If using ZIP (Option 2):** Make sure you manually added the complete `dataset/` folder with all the actual image files (not just empty folders)

---

### Problem: "Dataset not found" or "File not found"

**Solution:**
1. Check that the `dataset/` folder exists in your project folder
2. Make sure it has the correct structure (train/images, train/labels, etc.)
3. Open `dataset/data.yaml` and update the paths to match your computer's folder locations

---

### Problem: "python is not recognized"

**Solution:**
1. Install Python from https://www.python.org/downloads/
2. **Important:** Check "Add Python to PATH" during installation
3. Restart your computer
4. Open a new PowerShell window and try again

---

### Problem: Warning about "LF will be replaced by CRLF" for `.venv/` files

**What this means:** Git is trying to track your virtual environment folder, which it shouldn't.

**Solution:**
- This is usually harmless, but if it bothers you:
  1. Run: `git rm -r --cached .venv`
  2. Run: `git commit -m "Remove .venv from tracking"`
  3. The warnings will stop

---

### Problem: "No module named..." or import errors

**Solution:**
1. Make sure you activated the virtual environment (you should see `(.venv)` in your command line)
2. If not activated, run: `.\.venv\Scripts\Activate.ps1`
3. Make sure you installed requirements: `pip install -r requirements.txt`

---

### Problem: Program runs but finds no defects

**Possible reasons:**
1. The model hasn't been trained yet - train it first using `python train_yolo.py`
2. The confidence threshold is too high - try lowering it in the code
3. The images don't actually contain the types of defects the model was trained to find

---

## Understanding the Results

**What you'll see:**
- Images with colored boxes drawn around detected defects
- Each box has a label showing what type of defect was found
- A confidence score (0.0 to 1.0) showing how sure the program is
  - Higher numbers (closer to 1.0) = more confident
  - Lower numbers (closer to 0.0) = less confident

**Types of defects detected:**
- MultiByPassed - Multiple bypassed diodes
- MultiDiode - Multiple diode issues
- MultiHotSpot - Multiple hotspots
- SingleByPassed - Single bypassed diode
- SingleDiode - Single diode issue
- SingleHotSpot - Single hotspot
- StringOpenCircuit - Open circuit in a string
- StringReversedPolarity - Reversed polarity in a string

---

## Tips for Best Results

1. **Train the model first** - The program works much better with a trained model (`best.pt`) than with the default untrained model

2. **Use good quality images** - Clear thermal images work best

3. **Check the paths** - Make sure `dataset/data.yaml` has the correct paths for your computer

4. **Be patient** - Training takes time, but it's worth it for better results

5. **Start simple** - Try `python main.py` first to make sure everything works before training

---

## Need More Help?

- Check the individual README files in subfolders:
  - `webapp/README.md` - Web interface details
  - `hardware/README_hardware.md` - Camera setup
  - `software/README.md` - Command-line tool details
  - `dataset/README.md` - Dataset information

---

## Summary

**To get started quickly:**
1. Get the project files (Option 1 or 2 above)
2. Install Python
3. Set up the project (create virtual environment, install requirements)
4. Run: `python main.py` to test it

**For best results:**
1. Complete setup above
2. Train the model: `python train_yolo.py`
3. Use the trained model for predictions

The program will automatically use your trained model once it's created. Before training, it uses a basic model that works but isn't as accurate.
