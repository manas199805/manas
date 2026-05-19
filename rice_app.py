import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from PIL import Image
import urllib.parse

st.set_page_config(page_title="Rice Precision Pro (CG Custom Milling Edition)", layout="wide")
st.title("🌾 Rice Quality Analyzer & CG Procurement Compliance Engine")

# --- Sidebar Configuration ---
st.sidebar.header("Milling Parameters & Settings")

rice_variety = st.sidebar.selectbox(
    "Select Rice Variety", 
    ["Grade A (Long/Slender)", "Common (Medium/Short)"]
)

processing_type = st.sidebar.radio(
    "Milling Type (CG Govt / FCI FAQ Standards)",
    ["Raw Rice (Arva)", "Parboiled Rice (Usna)"]
)

phone_number = st.sidebar.text_input("WhatsApp Number", placeholder="e.g., 919876543210")
pixel_to_mm_val = st.sidebar.slider("Calibration (Pixel to MM)", 0.01, 0.20, 0.13)

GOVT_LIMITS = {
    "Raw Rice (Arva)": {"max_broken": 25.0, "max_small_broken": 1.0},
    "Parboiled Rice (Usna)": {"max_broken": 16.0, "max_small_broken": 1.0}
}

@st.cache_data(show_spinner=False)
def process_rice(img_array, variety, p2mm):
    # Keep a completely clean copy of the original user-uploaded image for drawing output
    display_img = img_array.copy()
    orig_h, orig_w = img_array.shape[:2]
    
    # Internal scaling for improving watershed threshold accuracy
    scale = 1.1
    w = int(orig_w * scale)
    h = int(orig_h * scale)
    img_resized = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    distance = ndimage.distance_transform_edt(thresh)
    coords = peak_local_max(distance, min_distance=15, labels=thresh)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=thresh)
    
    grain_data = []
    
    ratio_thresholds = {
        "Grade A (Long/Slender)": 2.5,
        "Common (Medium/Short)": 1.0
    }
    min_ratio = ratio_thresholds[variety]

    for label in np.unique(labels):
        if label == 0: 
            continue
            
        grain_mask = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            target_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(target_contour)
            if area < (80 * scale): # Scale the noise filter threshold dynamically
                continue 
            
            rect = cv2.minAreaRect(target_contour)
            
            # --- CORRECTION: Scale bounding metrics back down to fit original base image ---
            center_x, center_y = rect[0]
            dim1, dim2 = rect[1]
            angle = rect[2]
            
            orig_center = (center_x / scale, center_y / scale)
            orig_dims = (dim1 / scale, dim2 / scale)
            
            # Reconstruct the corrected box mapping coordinates
            corrected_rect = (orig_center, orig_dims, angle)
            box = np.intp(cv2.boxPoints(corrected_rect))
            
            # Calculate physical length using base p2mm factor
            length = max(orig_dims) * p2mm
            breadth = min(orig_dims) * p2mm
            
            current_ratio = length / breadth if breadth > 0 else 0
            if length > 5.0 and current_ratio < min_ratio:
                continue 

            grain_data.append({'length': length, 'box': box, 'center': orig_center})

    # --- Draw boxes and text natively on original dimension matrix ---
    for grain in grain_data:
        if grain['length'] < 1.5:
            color = (255, 0, 0)       # Red for Small Broken (RGB mapping adjustment)
        elif grain['length'] < 4.5:
            color = (255, 165, 0)     # Orange for Standard Broken
        else:
            color = (0, 255, 0)       # Green for Head Rice

        # Draw the tracking contour box
        cv2.drawContours(display_img, [grain['box']], 0, color, 2)
        
        # Safe text coordinates calculation (centered horizontally, shifted slightly up)
        tx = int(grain['center'][0] - 12)
        ty = int(grain['center'][1] + 4)
        
        # Bound check coordinates to ensure text is never dropped off-canvas
        tx = max(5, min(tx, orig_w - 30))
        ty = max(15, min(ty, orig_h - 5))
        
        # Draw background shadow text for heavy contrast/visibility
        cv2.putText(display_img, f"{grain['length']:.1f}", (tx, ty), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        # Draw primary text
        cv2.putText(display_img, f"{grain['length']:.1f}", (tx, ty), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    df = pd.DataFrame(grain_data)
    return df, display_img

uploaded_file = st.file_uploader("Upload Mill Sample Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with st.spinner("Analyzing milling quality..."):
        df, result_img = process_rice(np.array(image), rice_variety, pixel_to_mm_val)
    
    st.image(result_img, use_container_width=True)
    
    if not df.empty:
        total = len(df)
        avg_l = df['length'].mean()
        
        small_broken_count = (df['length'] < 1.5).sum()
        total_broken_count = (df['length'] < 4.5).sum()
        
        small_broken_pc = (small_broken_count / total) * 100
        total_broken_pc = (total_broken_count / total) * 100
        
        allowed_broken = GOVT_LIMITS[processing_type]["max_broken"]
        allowed_small_broken = GOVT_LIMITS[processing_type]["max_small_broken"]
        
        is_compliant = (total_broken_pc <= allowed_broken) and (small_broken_pc <= allowed_small_broken)
        
        st.subheader(f"📊 {rice_variety} ({processing_type}) Analysis Report")
        
        if is_compliant:
            st.success(f"✅ **Passes CG Government FAQ Norms:** Sample falls within the acceptable delivery standards.")
        else:
            st.error(f"❌ **Rejection Alert (FAQ Out of Bounds):** Sample exceeds maximum permissible government broken allowances.")
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Grains Counted", total)
        c2.metric("Average Grain Size", f"{avg_l:.2f} mm")
        c3.metric("Total Broken %", f"{total_broken_pc:.1f}%", f"Max Limit: {allowed_broken}%", delta_color="inverse")
        c4.metric("Small Broken %", f"{small_broken_pc:.1f}%", f"Max Limit: {allowed_small_broken}%", delta_color="inverse")
        
        st.markdown("### 📋 Procurement Specification Summary Table")
        comparison_data = {
            "Refraction Parameter": ["Total Broken Grains", "Small Broken Grains (Dust/Choor)"],
            "Current Sample Value": [f"{total_broken_pc:.2f}%", f"{small_broken_pc:.2f}%"],
            "Govt Target Cap (FAQ)": [f"Max {allowed_broken}%", f"Max {allowed_small_broken}%"],
            "Status": ["Pass" if total_broken_pc <= allowed_broken else "Reject", 
                       "Pass" if small_broken_pc <= allowed_small_broken else "Reject"]
        }
        st.table(pd.DataFrame(comparison_data))
        
        clean_phone = "".join(filter(str.isdigit, phone_number))
        if clean_phone:
            status_text = "PASSED (FAQ COMPLIANT)" if is_compliant else "REJECTED (EXCEEDS LIMITS)"
            msg = f"*{rice_variety} - {processing_type} REPORT*\n\nStatus: {status_text}\nTotal Grains: {total}\nAvg Size: {avg_l:.2f}mm\nTotal Broken: {total_broken_pc:.1f}% (Max: {allowed_broken}%)\nSmall Broken: {small_broken_pc:.1f}% (Max: {allowed_small_broken}%)"
            url = f"https://wa.me/{clean_phone}?text={urllib.parse.quote(msg)}"
            
            st.markdown(
                f'''<a href="{url}" target="_blank" style="text-decoration: none;">
                    <button style="background-color:#25D366; color:white; padding:12px 20px; 
                    font-weight:bold; border-radius:8px; border:none; cursor:pointer; width:100%;">
                        📲 Send Procurement Report to WhatsApp
                    </button>
                </a>''', 
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ No rice grains detected. Please adjust the threshold or check image lighting.")
