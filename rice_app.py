import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from PIL import Image
import urllib.parse

st.set_page_config(page_title="Rice Precision Scanner", layout="wide")
st.title("🌾 Every-Grain Analysis (BIS Standards)")

# --- Sidebar ---
st.sidebar.header("Scanner Settings")
phone_number = st.sidebar.text_input("WhatsApp Number (e.g., 919876543210)")
pixel_to_mm = st.sidebar.slider("Calibration (Pixel to MM)", 0.01, 0.20, 0.13)

def process_rice(img_array):
    # Virtual Pixel Density Increase (Scaling)
    scale = 1.2
    w = int(img_array.shape[1] * scale)
    h = int(img_array.shape[0] * scale)
    img = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    display_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Precise Thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 21, 5)
    
    # Watershed to separate every single grain
    distance = ndimage.distance_transform_edt(thresh)
    coords = peak_local_max(distance, min_distance=12, labels=thresh)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=thresh)
    
    grain_data = []
    adj_ratio = pixel_to_mm / scale
    
    for label in np.unique(labels):
        if label == 0: continue
        grain_mask = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            area = cv2.contourArea(contours[0])
            if area < 40: continue 
            
            rect = cv2.minAreaRect(contours[0])
            box = np.intp(cv2.boxPoints(rect))
            length = max(rect[1]) * adj_ratio
            center = rect[0]
            
            grain_data.append({'length': length, 'box': box, 'center': center})

            # DISPLAY: Write size on EVERY grain
            color = (0, 255, 0) if length >= 4.0 else (255, 0, 0)
            cv2.drawContours(display_img, [box], 0, color, 2)
            
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
        st.image(result_img, use_container_width=True)
        
        total = len(df)
        avg_l = df['length'].mean()
        broken_pc = (len(df[df['length'] < 4.0]) / total) * 100
        
        # Report
        st.subheader("📊 Analysis Report")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Grains", total)
        c2.metric("Avg Size", f"{avg_l:.2f} mm")
        c3.metric("Broken (%)", f"{broken_pc:.1f}%")
        
        # WhatsApp Share
        if phone_number:
            msg = f"RICE REPORT\nTotal Grains: {total}\nAvg Size: {avg_l:.2f}mm\nBroken: {broken_pc:.1f}%"
            whatsapp_url = f"https://wa.me/{phone_number}?text={urllib.parse.quote(msg)}"
            st.markdown(f'''<a href="{whatsapp_url}" target="_blank">
                <button style="background-color:#25D366;color:white;padding:10px;border-radius:5px;border:none;cursor:pointer;width:100%;">
                📲 Share to WhatsApp
                </button></a>''', unsafe_allow_html=True)