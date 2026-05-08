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


uploaded_files = st.file_uploader(
    "Upload panoramic X-ray images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="uploader"
)


if uploaded_files:
    existing_names = [
        f.name for f in st.session_state.uploaded_images
    ]

    for file in uploaded_files:
        if file.name not in existing_names:
            st.session_state.uploaded_images.append(file)


st.divider()


col1, col2 = st.columns([1, 1])

with col1:
    run_prediction = st.button(
        "Run Segmentation",
        use_container_width=True,
    )

with col2:
    clear_all = st.button(
        "Remove All Images",
        use_container_width=True,
    )
if run_prediction:

    st.write("BUTTON PRESSED")

    if len(st.session_state.uploaded_images) == 0:

        st.warning("Please upload images first")

    else:

        files = []

        for file in st.session_state.uploaded_images:

            file.seek(0)

            files.append(
                (
                    "files",
                    (
                        file.name,
                        file.read(),
                        file.type,
                    ),
                )
            )

        with st.spinner("Running segmentation..."):

            try:

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=60,
                )

                st.write("STATUS:", response.status_code)

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
    st.rerun()

if len(st.session_state.uploaded_images) > 0:

    st.divider()

    st.subheader("Results Viewer")

    filenames = [f.name for f in st.session_state.uploaded_images]

    selected_index = st.selectbox(
        "Select image",
        options=list(range(len(filenames))),
        format_func=lambda x: filenames[x],
        key="selector",
    )

    st.session_state.current_index = selected_index

    col1, col2 = st.columns(2)

    current_file = st.session_state.uploaded_images[selected_index]

    with col1:
        st.markdown("### Original")
        st.image(Image.open(current_file), use_container_width=True)

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

if len(st.session_state.uploaded_images) > 0:

    st.divider()
    st.subheader("Manage Images")

    # list of names (stable)
    names = [f.name for f in st.session_state.uploaded_images]

    selected_index = st.selectbox(
        "Select image to remove",
        options=list(range(len(names))),
        format_func=lambda i: names[i],
        key="delete_selector",
    )

    if st.button("Remove Selected Image", key="remove_btn"):

        selected_name = names[selected_index]


        st.session_state.uploaded_images = [
            f for f in st.session_state.uploaded_images
            if f.name != selected_name
        ]


        st.session_state.predictions = [
            p for p in st.session_state.predictions
            if p["filename"] != selected_name
        ]

        st.session_state.current_index = 0

        st.rerun()