import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from PIL import Image
import urllib.parse

st.set_page_config(page_title="Rice Quality Analyzer", layout="wide")
st.title("🌾 Rice Analysis & Average Length Report")

# --- Settings ---
st.sidebar.header("Parameters")
phone_number = st.sidebar.text_input("WhatsApp Number (with country code)")
pixel_to_mm = st.sidebar.slider("Calibration (Pixel to MM)", 0.01, 0.20, 0.13)

def process_rice(img_array):
    # Image Prep
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # STRICT THRESHOLD: Ignores background, only picks up bright white rice
    _, thresh = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Separate grains
    distance = ndimage.distance_transform_edt(thresh)
    coords = peak_local_max(distance, min_distance=15, labels=thresh)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=thresh)
    
    grain_data = []
    display_img = img_array.copy()

    for label in np.unique(labels):
        if label == 0: continue
        grain_mask = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            area = cv2.contourArea(contours[0])
            if area < 100: continue 
            
            rect = cv2.minAreaRect(contours[0])
            box = np.intp(cv2.boxPoints(rect))
            length = max(rect[1]) * pixel_to_mm
            center = rect[0]
            
            grain_data.append({'length': length})

            # Drawing on image
            color = (0, 255, 0) if length >= 4.0 else (255, 0, 0)
            cv2.drawContours(display_img, [box], 0, color, 2)
            cv2.putText(display_img, f"{length:.1f}", (int(center[0]-10), int(center[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    df = pd.DataFrame(grain_data)
    return df, display_img

uploaded_file = st.file_uploader("Upload Rice Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    df, result_img = process_rice(np.array(image))
    
    if df is not None and not df.empty:
        st.image(result_img, use_container_width=True)
        
        # --- AVERAGE LENGTH ANALYSIS ---
        total_grains = len(df)
        avg_length = df['length'].mean()
        max_length = df['length'].max()
        min_length = df['length'].min()
        broken_pc = (len(df[df['length'] < 4.0]) / total_grains) * 100
        
        st.subheader("📊 Average Length Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Avg Length", f"{avg_length:.2f} mm")
        col2.metric("Max Length", f"{max_length:.2f} mm")
        col3.metric("Min Length", f"{min_length:.2f} mm")
        col4.metric("Total Grains", total_grains)
        
        st.write(f"**Broken Percentage:** {broken_pc:.1f}%")
        st.write(f"**Estimated HeadRice:** {(100 - broken_pc):.1f}%")
        
        # WhatsApp logic
        if phone_number:
            msg = f"*RICE ANALYSIS REPORT*\nAvg Length: {avg_length:.2f}mm\nBroken: {broken_pc:.1f}%\nTotal Grains: {total_grains}"
            url = f"https://wa.me/{phone_number}?text={urllib.parse.quote(msg)}"
            st.markdown(f'<a href="{url}" target="_blank"><button style="background-color:#25D366;color:white;padding:10px;border-radius:5px;border:none;cursor:pointer;">📲 Send Report to WhatsApp</button></a>', unsafe_allow_html=True)
    else:
        st.warning("No rice detected. Check lighting or background contrast.")
