import streamlit as st

# set tile
st.title("Ask a question to the image")

# set header
st.header("This app will help you ask questions about an image. Plesae upload an Image")

# upload file
file = st.file_uploader("" , type=["jpeg", "jpg" , "png"])
if file :
    # dipslay image
    st.image(file , use_column_width=True)

    # text input
    user_question = st.text_input("Ask a question about your image")

    # Agent response 
    if user_question and user_question!="" :
        st.write("dummy response ")
