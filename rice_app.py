import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from PIL import Image
import urllib.parse  # For web linking

st.set_page_config(page_title="Rice Analysis Pro", layout="wide")
st.title("🌾 Rice Quality Analyzer + WhatsApp Sharing")

# --- Sidebar Settings ---
st.sidebar.header("WhatsApp Settings")
phone_number = st.sidebar.text_input("Receiver Phone (with Country Code)", placeholder="919876543210")
pixel_to_mm = st.sidebar.slider("Calibration (Pixel to MM)", 0.01, 0.20, 0.13)

def process_rice(img_array):
    # (High-precision scanning logic)
    scale_percent = 150 
    w_new = int(img_array.shape[1] * scale_percent / 100)…
[10:57, 19/05/2026] Manas Agrawal: import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from PIL import Image
import urllib.parse

st.set_page_config(page_title="Rice Grain Precision Scanner", layout="wide")
st.title("🌾 Every-Grain Precision Analysis")

# --- Settings ---
st.sidebar.header("Scanner Settings")
phone_number = st.sidebar.text_input("WhatsApp Number (with Country Code)", placeholder="91...")
pixel_to_mm = st.sidebar.slider("Calibration (Pixel to MM)", 0.01, 0.20, 0.13)

def process_rice(img_array):
    # Image Upscaling for better edge detection
    scale = 1.2
    w = int(img_array.shape[1] * scale)
    h = int(img_array.shape[0] * scale)
    img = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    display_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Adaptive threshold to find every single grain boundary
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 21, 5)
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Watershed to separate touching grains accurately
    distance = ndimage.distance_transform_edt(thresh)
    coords = peak_local_max(distance, min_distance=12, labels=thresh)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=thresh)
    
    grain_data = []
    # Adjust ratio for the upscale
    adj_ratio = pixel_to_mm / scale
    
    for label in np.unique(labels):
        if label == 0: continue
        grain_mask = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            area = cv2.contourArea(contours[0])
            if area < 40: continue # Ignore tiny dust
            
            rect = cv2.minAreaRect(contours[0])
            box = np.intp(cv2.boxPoints(rect))
            
            # Accurate length measurement
            length = max(rect[1]) * adj_ratio
            breadth = min(rect[1]) * adj_ratio
            center = rect[0]
            
            grain_data.append({'length': length, 'breadth': breadth, 'box': box, 'center': center})

            # --- Write size on every single grain ---
            # Color coding: Green for Sound, Red for Broken (< 4.0mm)
            color = (0, 255, 0) if length >= 4.0 else (255, 0, 0)
            
            cv2.drawContours(display_img, [box], 0, color, 2)
            
            # Text placement at the center of the grain
            text = f"{length:.1f}"
            cv2.putText(display_img, text, (int(center[0]-15), int(center[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    df = pd.DataFrame(grain_data)
    return df, display_img

uploaded_file = st.file_uploader("Upload Rice Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    df, result_img = process_rice(np.array(image))
    
    if df is not None:
        st.image(result_img, use_container_width=True, caption="Detailed Grain Analysis")
        
        # Calculations for Report
        total = len(df)
        avg_l = df['length'].mean()
        broken_count = len(df[df['length'] < 4.0])
        broken_pc = (broken_count / total) * 100
        
        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Grains Found", total)
        m2.metric("Average Size", f"{avg_l:.2f} mm")
        m3.metric("Broken Percentage", f"{broken_pc:.1f}%")
        
        # WhatsApp Feature
        if phone_number:
            msg = f"RICE QUALITY REPORT\nTotal Grains: {total}\nAvg Size: {avg_l:.2f}mm\nBroken: {broken_pc:.1f}%"
            whatsapp_url = f"https://wa.me/{phone_number}?text={urllib.parse.quote(msg)}"
            st.markdown(f'<a href="{whatsapp_url}" target="_blank"><button style="background-color:#25D366;color:white;padding:10px;border-radius:5px;border:none;cursor:pointer;">📲 Send to WhatsApp</button></a>', unsafe_allow_html=True)
