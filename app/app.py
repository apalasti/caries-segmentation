import base64
import io

import requests
import streamlit as st
from PIL import Image


API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Dental Caries Segmentation",
    layout="wide",
)

st.title("Dental Caries Segmentation UI")


if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

if "predictions" not in st.session_state:
    st.session_state.predictions = []

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "upload_lock" not in st.session_state:
    st.session_state.upload_lock = False

if "ui_version" not in st.session_state:
    st.session_state.ui_version = 0


uploaded_files = st.file_uploader(
    "Upload panoramic X-ray images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.ui_version}"
)

if uploaded_files and not st.session_state.upload_lock:

    existing_names = {
        f["name"] for f in st.session_state.uploaded_images
    }

    for file in uploaded_files:

        if file.name not in existing_names:

            st.session_state.uploaded_images.append({
                "name": file.name,
                "type": file.type,
                "bytes": file.read(),
            })

    st.session_state.upload_lock = True


st.divider()


col1, col2 = st.columns(2)

with col1:
    run_prediction = st.button("Run Segmentation", use_container_width=True)

with col2:
    clear_all = st.button("Remove All Images", use_container_width=True)


if run_prediction:

    if not st.session_state.uploaded_images:
        st.warning("Please upload images first")

    else:

        files = [
            ("files", (f["name"], f["bytes"], f["type"]))
            for f in st.session_state.uploaded_images
        ]

        with st.spinner("Running segmentation..."):

            try:
                response = requests.post(API_URL, files=files, timeout=60)

                if response.status_code == 200:
                    st.session_state.predictions = response.json()["results"]
                    st.success("Prediction complete")
                else:
                    st.error(response.text)

            except Exception as e:
                st.error(str(e))


if clear_all:

    st.session_state.uploaded_images = []
    st.session_state.predictions = []
    st.session_state.current_index = 0
    st.session_state.upload_lock = False

    st.session_state.ui_version += 1

    st.rerun()


if st.session_state.uploaded_images:

    st.divider()
    st.subheader("Results Viewer")

    filenames = [f["name"] for f in st.session_state.uploaded_images]

    selected_index = st.selectbox(
        "Select image",
        options=list(range(len(filenames))),
        format_func=lambda i: filenames[i],
        key=f"selector_{st.session_state.ui_version}",
    )

    current_file = st.session_state.uploaded_images[selected_index]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original")
        img = Image.open(io.BytesIO(current_file["bytes"]))
        st.image(img, use_container_width=True)

    with col2:

        if len(st.session_state.predictions) > selected_index:

            pred = st.session_state.predictions[selected_index]

            overlay_bytes = base64.b64decode(pred["overlay"])
            overlay_img = Image.open(io.BytesIO(overlay_bytes))

            st.markdown("### Segmentation")
            st.markdown(f"Model: {pred['model']}")
            st.image(overlay_img, use_container_width=True)

        else:
            st.info("Run segmentation first")



if st.session_state.uploaded_images:

    st.divider()
    st.subheader("Manage Images")

    names = [f["name"] for f in st.session_state.uploaded_images]

    selected_index = st.selectbox(
        "Select image to remove",
        options=list(range(len(names))),
        format_func=lambda i: names[i],
        key=f"delete_selector_{st.session_state.ui_version}",
    )

    if st.button("Remove Selected Image", key="remove_btn"):

        print(
            "UPLOADED FILES: ",uploaded_files)
        selected_name = names[selected_index]


        # remove image
        st.session_state.uploaded_images = [
            f for f in st.session_state.uploaded_images
            if f["name"] != selected_name
        ]

        # remove prediction
        st.session_state.predictions = [
            p for p in st.session_state.predictions
            if p["filename"] != selected_name
        ]

        st.session_state.upload_lock = False

        # force full UI refresh
        st.session_state.ui_version += 1

        st.rerun()