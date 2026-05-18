import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from PIL import Image

st.set_page_config(page_title="Rice Analysis App", layout="wide")
st.title("🌾 Rice Quality Analyzer (BIS Standards)")

st.sidebar.header("Settings")
pixel_to_mm = st.sidebar.slider("Calibration (Pixel to MM)", 0.01, 0.20, 0.13)

def process_rice(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape
    
    # ROI: Crop 10% from edges to ignore shadows/page borders
    crop_h, crop_w = int(h * 0.1), int(w * 0.1)
    img = img[crop_h:h-crop_h, crop_w:w-crop_w]
    display_img = img_array[crop_h:h-crop_h, crop_w:w-crop_w].copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. INTENSITY FILTER: Only detect objects brighter than the background
    # This prevents the blue surface from being scanned as rice
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY) 
    
    # Clean up small noise dots
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Watershed to separate touching grains
    distance = ndimage.distance_transform_edt(thresh)
    coords = peak_local_max(distance, min_distance=10, labels=thresh)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=thresh)
    
    grain_data = []
    for label in np.unique(labels):
        if label == 0: continue
        grain_mask = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        
        area = cv2.contourArea(contours[0])
        # SIZE FILTER: Ignore tiny specks and huge clumps
        if area < 120 or area > 5000: continue 

        rect = cv2.minAreaRect(contours[0])
        box = np.intp(cv2.boxPoints(rect))
        L = max(rect[1]) * pixel_to_mm
        B = min(rect[1]) * pixel_to_mm
        
        # 2. SLENDERNESS FILTER: Rice is long, background noise is usually square
        if B == 0 or (L / B) < 1.3: continue 

        grain_data.append({'length': L, 'breadth': B, 'box': box})

    df = pd.DataFrame(grain_data)
    if df.empty: return None, None

    # BIS Classification (Below 4.0mm is broken)
    def classify(l):
        if l >= 5.0: return "Sound Kernal"
        elif 4.0 <= l < 5.0: return "Above Broken Size"
        elif 3.0 <= l < 4.0: return "Big Broken"
        elif 2.0 <= l < 3.0: return "Medium Broken"
        else: return "Small Broken"

    df['category'] = df['length'].apply(classify)
    
    colors = {
        "Sound Kernal": (0, 255, 0), "Above Broken Size": (255, 0, 0),
        "Big Broken": (255, 255, 0), "Medium Broken": (255, 165, 0),
        "Small Broken": (255, 0, 0)
    }
    for i, row in df.iterrows():
        cv2.drawContours(display_img, [row['box']], 0, colors.get(row['category'], (255,255,255)), 2)

    return df, display_img

uploaded_file = st.file_uploader("Upload Rice Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    df, result_img = process_rice(img_array)
    
    if df is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(result_img, use_column_width=True)
        with col2:
            st.subheader("📊 BIS Analysis Report")
            total = len(df)
            broken_df = df[df['length'] < 4.0] # BIS Broken Rule
            broken_pc = (len(broken_df) / total) * 100
            
            st.metric("Total Grains", total)
            st.metric("Avg Length", f"{df['length'].mean():.2f} mm")
            st.metric("Total Broken (%)", f"{broken_pc:.1f}%")
            st.metric("HeadRice (%)", f"{(100 - broken_pc):.1f}%")
            
            st.write("---")
            st.write("*Broken Breakdown:*")
            counts = df['category'].value_counts(normalize=True) * 100
            st.write(f"🔹 Big Broken: {counts.get('Big Broken', 0):.1f}%")
            st.write(f"🔹 Medium Broken: {counts.get('Medium Broken', 0):.1f}%")
            st.write(f"🔹 Small Broken: {counts.get('Small Broken', 0):.1f}%")