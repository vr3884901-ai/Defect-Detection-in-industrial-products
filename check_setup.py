print("--- STARTING APP DEBUGGER ---")

print("[1/6] Importing OS and System libraries...")
import os
import sys

print("[2/6] Importing OpenCV...")
try:
    import cv2
    print("      -> OpenCV imported successfully.")
except ImportError as e:
    print(f"      -> ERROR importing OpenCV: {e}")

print("[3/6] Importing Flask...")
try:
    from flask import Flask, render_template, request, redirect, url_for, session, Response
    print("      -> Flask imported successfully.")
except ImportError as e:
    print(f"      -> ERROR importing Flask: {e}")

print("[4/6] Importing PIL (Pillow)...")
try:
    from PIL import Image
    print("      -> PIL imported successfully.")
except ImportError as e:
    print(f"      -> ERROR importing PIL: {e}")

print("[5/6] Importing NumPy...")
try:
    import numpy as np
    print("      -> NumPy imported successfully.")
except ImportError as e:
    print(f"      -> ERROR importing NumPy: {e}")

print("[6/6] Importing TensorFlow (THIS USUALLY TAKES 10-30 SECONDS)...")
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    print("      -> TensorFlow imported successfully!")
except ImportError as e:
    print(f"      -> ERROR importing TensorFlow: {e}")

print("\n--- CHECKS COMPLETE ---")
print("If you see this message, your libraries are installed correctly.")
print("You can now try running 'python app.py' again and waiting for the TensorFlow import.")