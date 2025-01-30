import streamlit as st


def page_architecture():
    st.title("System Architecture")

    # Provide the path to your PNG image
    image_path = "./assets/quant-dev 2.svg"

    try:
        # Display the image in Streamlit
        st.image(
            image_path, caption="System Architecture Diagram", use_container_width=True
        )
    except Exception as e:
        st.error(f"Error loading image: {e}")


# If running the script directly, show the page
if __name__ == "__main__":
    page_architecture()
