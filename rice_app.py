import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from PIL import Image
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import urllib.parse

st.set_page_config(page_title="Rice Analysis Pro", layout="wide")

def create_pdf(df, result_img):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, h - 50, "Rice Quality Analysis Report")
    p.setFont("Helvetica", 12)
    p.drawString(50, h - 80, f"Total Grains: {len(df)}")
    p.drawString(50, h - 100, f"Avg Length: {df['length'].mean():.2f} mm")
    p.drawString(50, h - 120, f"Broken Percentage: {(len(df[df['length'] < 4.0])/len(df))*100:.1f}%")
    
    img_pil = Image.fromarray(result_img)
    img_buffer = io.BytesIO()
    img_pil.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    p.drawInlineImage(img_pil, 50, h - 450, width=500, preserveAspectRatio=True)
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

def process_rice(img_array, pixel_to_mm):
    # Convert and Grayscale
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # IMPROVED SCANNING: Otsu's thresholding finds the whole grain automatically
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Fill small holes inside the grains for better accuracy
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Separate touching grains
    distance = ndimage.distance_transform_edt(thresh)
    coords = peak_local_max(distance, min_distance=10, labels=thresh)
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
            if area < 50: continue 
            
            rect = cv2.minAreaRect(contours[0])
            box = np.intp(cv2.boxPoints(rect))
            length = max(rect[1]) * pixel_to_mm
            
            grain_data.append({'length': length, 'box': box, 'center': rect[0]})
            color = (0, 255, 0) if length >= 4.0 else (255, 0, 0)
            cv2.drawContours(display_img, [box], 0, color, 2)
            cv2.putText(display_img, f"{length:.1f}", (int(rect[0][0]), int(rect[0][1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return pd.DataFrame(grain_data), display_img

# --- UI ---
st.title("🌾 Rice Quality Analyzer")
pixel_to_mm = st.sidebar.slider("Calibration", 0.01, 0.20, 0.13)
uploaded_file = st.file_uploader("Upload Rice Image", type=['jpg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file)
    df, res_img = process_rice(np.array(img), pixel_to_mm)
    
    if not df.empty:
        st.image(res_img, use_container_width=True)
        st.metric("Total Grains", len(df))
        st.metric("Avg Length", f"{df['length'].mean():.2f} mm")
        
        # PDF Option
        pdf_file = create_pdf(df, res_img)
        st.download_button("📥 Download PDF Report", pdf_file, "Rice_Report.pdf", "application/pdf")
