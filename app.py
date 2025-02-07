'''
This is the main file for the Streamlit app. A wood cross section image is uploaded and the app will display the results of the DeepCS-TRD model.
In the image, the pith pixel is marked interactively by the user. The app will then display the results of the DeepCS-TRD model.
'''
import streamlit as st
import numpy as np
import cv2
import os

from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from pathlib import Path

from urudendro.image import load_image
from deep_cstrd.deep_tree_ring_detection import DeepTreeRingDetection
from cross_section_tree_ring_detection.cross_section_tree_ring_detection import saving_results

class DeepCSTRD_MODELS:
    pinus_v1 = "Pinus V1"
    pinus_v2 = "Pinus V2"
    gleditsia = "Gleditsia"
    salix = "Salix Glauca"

def get_model_path(model_id):
    root_path = "./models/deep_cstrd/"
    if model_id == DeepCSTRD_MODELS.pinus_v1:
        return os.path.join(root_path, "256_pinus_v1_1504.pth")
    elif model_id == DeepCSTRD_MODELS.pinus_v2:
        return os.path.join(root_path, "256_pinus_v2_1504.pth")
    elif model_id == DeepCSTRD_MODELS.gleditsia:
        return os.path.join(root_path, "256_gleditsia_1504.pth")
    elif model_id == DeepCSTRD_MODELS.salix:
        return os.path.join(root_path, "256_salix_1504.pth")
    else:
        raise "models does not exist"

# Reduce app margins
run = False
st.set_page_config(layout="wide")
output_dir = Path("./output/app")
# Application configuration
st.title("DeepCS-TRD, a Deep Learning-based Cross-Section Tree Ring Detector")
st.write("Upload an image and click to mark the pith of the wood disk.")

#add check box to displya parameters
st.sidebar.write("DeepCS-TRD Parameters")
check = st.sidebar.checkbox("Show Parameters", value=False)
if check:
    # Adjustable parameters
    alpha = st.slider("Angle α (degrees)", 0, 90, 30, 5)
    tile_size = st.selectbox("Tile Size", [0, 64, 128, 256, 512], index=3)
    prediction_map_threshold = st.slider("Prediction Map Threshold", 0.0, 1.0, 0.2, 0.1)
    total_rotations = st.slider("Total Rotations", 0, 8, 4, 1)
    hsize = st.slider("Height", 0, 3500, 1504, 100)
    wsize = st.slider("Width", 0, 3500, 1504, 100)
else:
    alpha = 30
    tile_size = 256
    prediction_map_threshold = 0.2
    total_rotations = 4
    hsize = 1504
    wsize = 1504

st.divider()
# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
st.divider()
if uploaded_file:
    image = Image.open(uploaded_file)
    original_size = image.size  # Save original size
    image_orig = np.array(image)

    # Add sliders to adjust image size
    scale = st.slider("Scale image", 0.1, 1.0, 0.3, 0.1)
    new_size = (int(image_orig.shape[1] * scale), int(image_orig.shape[0] * scale))
    image = Image.fromarray(image_orig).resize(new_size)
    image = np.array(image)

    # Display interactive image
    st.warning("Click on the image to mark the pith.")
    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:
        coords = streamlit_image_coordinates(image)

    if coords:
        x, y = coords["x"], coords["y"]
        cx = int(x / scale)
        cy = int(y / scale)
        st.session_state["coords"] = (cx, cy)
        st.write(f"Last selected position in original scale: X = {cx}, Y = {cy}")
        model_radio  = st.radio("Select a model", [DeepCSTRD_MODELS.pinus_v1, DeepCSTRD_MODELS.pinus_v2,  DeepCSTRD_MODELS.gleditsia, DeepCSTRD_MODELS.salix], horizontal=True)
        if st.button("Run"):
            st.warning("Running DeepCS-TRD...")
            os.system(f"rm -rf {output_dir}")
            output_dir.mkdir(exist_ok=True, parents=True)
            input_path = str( output_dir / "input.png")

            cv2.imwrite(str(input_path), cv2.cvtColor(image_orig, cv2.COLOR_RGB2BGR))
            img_in  = load_image(input_path)
            sigma = 3
            th_low = 5
            th_high = 20
            nr = 360
            min_chain_length = 2
            weights_path = get_model_path(model_radio)
            res = DeepTreeRingDetection(img_in, int(cy), int(cx), sigma, th_low, th_high, hsize,
                                        wsize,
                                        alpha, nr, min_chain_length, weights_path,
                                        total_rotations,
                                        False, input_path, output_dir, tile_size,
                                        prediction_map_threshold)

            saving_results(res, output_dir, True)
            st.warning("DeepCS-TRD finished.")


st.divider()
uploaded_files = ["chains.png", "connect.png","postprocessing.png", "output.png"]
if "current_image_index" not in st.session_state:
    st.session_state["current_image_index"] = 0

# Display the current image
current_image_path = uploaded_files[st.session_state["current_image_index"]]
if (output_dir / current_image_path).exists():
    current_image = Image.open(str(output_dir / current_image_path))


    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write(Path(current_image_path).stem)
        st.image(current_image, caption=current_image_path, use_column_width=True)

    with col1:
        if st.button("Next"):
            st.session_state["current_image_index"] = (st.session_state["current_image_index"] + 1) % len(
                uploaded_files)

        if st.button("Previous"):
            st.session_state["current_image_index"] = (st.session_state["current_image_index"] - 1) % len(
                uploaded_files)








