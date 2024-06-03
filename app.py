from pathlib import Path
from PIL import Image
import streamlit as st

import numpy as np
from processing_engine import ProcessingEngine


def process_video(video_path, engine, grid=False):
    st.write(":orange[Processing video..]")
    snippets_path, embedding_index_path = engine.process_video(video_path, grid=grid)
    st.write(
        f"Done!! :sunglasses: :green[Wrote results to {snippets_path} and {embedding_index_path}]"
    )
    return (snippets_path, embedding_index_path)


def get_processed_video_snippes_and_index_paths(video_path):
    video_path = Path(video_path)
    video_dir = video_path.parent
    base_filename = video_path.stem

    snippets_path = video_dir / f"{base_filename}_snippets"
    index_path = video_dir / f"{base_filename}_faiss.index"
    return (snippets_path, index_path)


def process_image(
    image_path, engine, snippets_path, embedding_index_path, num_results_to_show=10
):
    st.write(":green[Processing query image..]")
    embedding = engine.process_image(image_path)
    query = np.expand_dims(embedding, axis=0)
    st.write("Done computing embedding !! :yellow[matching with images now..]")
    if snippets_path is None or embedding_index_path is None:
        st.write("No index specified!")
        return
    K = 100
    matching_indices, _ = engine.search_embeddings(query, embedding_index_path, k=K)
    st.write(":blue[Matching snippet images:]")
    cols = 2
    image_cols = []
    for _ in range(cols):
        image_cols.append([])

    for i in range(num_results_to_show):
        idx = matching_indices[i]
        search_pattern = f"{idx}.*"
        snippet_path = next(snippets_path.glob(search_pattern))
        snippet_image = Image.open(snippet_path)
        image_cols[i % cols].append(snippet_image)

    for col in image_cols:
        st.image(col, caption=None, width=200)

    st.write("Done!!!! :sunglasses:")


def get_video_files(uploads_dir):
    return sorted(
        list(uploads_dir.glob("*.mp4"))
        + list(uploads_dir.glob("*.avi"))
        + list(uploads_dir.glob("*.mov"))
    )


def get_image_files(uploads_dir):
    return sorted(
        list(uploads_dir.glob("*.jpg"))
        + list(uploads_dir.glob("*.png"))
        + list(uploads_dir.glob("*.jpeg"))
    )


def main():
    st.title("Triton Demo App")
    triton_engine = ProcessingEngine()

    tabs = st.tabs(["Datasets", "Identities", "Upload and Process", "Run Searches"])
    with tabs[0]:
        st.header("Datasets")
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        video_files = get_video_files(uploads_dir)

        selected_video = st.selectbox(
            "Select a processed dataset/video to view", [None] + video_files
        )
        if selected_video is not None:
            with open(selected_video, "rb") as video_fp:
                st.video(video_fp.read())

    with tabs[1]:
        st.header("Identities")
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        image_files = get_image_files(uploads_dir)

        selected_image = st.selectbox(
            "Select an identity to view", [None] + image_files
        )
        if selected_image is not None:
            with open(selected_image, "rb") as image_fp:
                st.image(image_fp.read())

    with tabs[2]:
        st.header("Upload and Process")

        # =============================
        # Upload video and trigger processing
        # =============================
        grid = st.checkbox(
            "Use a grid to derive snippets from image (:red[experimental and slow])"
        )
        uploaded_file = st.file_uploader(
            "Upload a video to process - \
                :blue[generate snippets using detected persons or full image grid]",
            type=["mp4", "avi", "mov"],
        )

        snippets_path, embedding_index_path = None, None

        if uploaded_file is not None:
            # Ensure the 'uploads' directory exists
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(parents=True, exist_ok=True)

            # Save the uploaded file to a temporary directory
            video_path = uploads_dir / uploaded_file.name
            if Path(video_path).exists():
                st.write(f"{video_path} already exists, skipping re-processing")
            else:
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Call the function to process the uploaded video
                snippets_path, embedding_index_path = process_video(
                    video_path, engine=triton_engine, grid=grid
                )

            # show the video
            video_bytes = uploaded_file.read()
            st.video(video_bytes)

        # =============================
        # Upload image identity
        # =============================
        uploaded_file = None
        uploaded_file = st.file_uploader(
            "Upload a query image", type=["jpg", "png", "jpeg"]
        )

        if uploaded_file is not None:
            # Ensure the 'uploads' directory exists (it should though from
            # above)
            uploads_dir = Path("uploads")
            uploads_dir.mkdir(parents=True, exist_ok=True)

            # Save the uploaded file to a temporary directory
            image_path = uploads_dir / uploaded_file.name
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # show the image
            image_bytes = uploaded_file.read()
            st.image(image_bytes)

    with tabs[3]:
        # =============================
        # Choose processed video and identity and run search
        # =============================
        st.header("Run Searches")

        num_results_to_show = st.slider("Number of matches to show", 0, 100, 10, 5)

        uploads_dir = Path("uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # List uploaded videos
        video_files = get_video_files(uploads_dir)
        selected_video = st.selectbox("Select a video", [None] + video_files)

        snippets_path, embedding_index_path = None, None
        if selected_video:
            snippets_path, embedding_index_path = (
                get_processed_video_snippes_and_index_paths(selected_video)
            )

        # List uploaded query images
        image_files = get_image_files(uploads_dir)
        selected_image = st.selectbox("Select a query image", [None] + image_files)

        if selected_image and snippets_path and embedding_index_path:
            # Show selected image
            selected_image_path = Path(selected_image)
            st.image(Image.open(selected_image_path))

            # Process and search with the selected query image
            process_image(
                selected_image_path,
                triton_engine,
                snippets_path,
                embedding_index_path,
                num_results_to_show=num_results_to_show,
            )


if __name__ == "__main__":
    main()
