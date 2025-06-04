import streamlit as st
from utils import (
    classify_sign_cnn,
    classify_sign_mobilenet,
    classify_sign_resnet50_pytorch,
    classify_sign_efficientnet
)

### ustawienie zdjęcia w tle
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://raw.githubusercontent.com/ddomsondd/asl_atlas/main/background_pic.jpg");
    background-size: contain;
    background-position: center;
    background-attachment: fixed;
}
</style>
'''

# Wstawienie CSS do aplikacji
st.markdown(page_bg_img, unsafe_allow_html=True)

###

st.title("American Sign Language Recognition")


st.sidebar.title("Wybierz model")
option = st.sidebar.selectbox("Wybierz opcje", ["CNN", "MobileNet", "ResNet50 (PyTorch)", "EfficientNet (PyTorch)"])

if option == "CNN":
    st.write("MODEL CNN")
    uploaded_file = st.file_uploader("Wprowadź zdjęcie")
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        col1.image(uploaded_file)

        prediction = classify_sign_cnn(uploaded_file)
        col2.write(f"Predicted class: {prediction}")

        true_sign = col2.text_input("Podaj jaki znak wprowadziłeś")
        if true_sign:
            if true_sign == prediction:
                st.balloons()
                col2.success(f"Znak został prawidłowo sklasyfikowany! :D")
            else:
                col2.error("Niestety tym razem się nie udało :(")


elif option == "MobileNet":
    st.write("MODEL MobileNet")

    uploaded_file = st.file_uploader("Wprowadź zdjęcie")
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        col1.image(uploaded_file)

        prediction = classify_sign_mobilenet(uploaded_file)
        col2.write(f"Predicted class: {prediction}")

        true_sign = col2.text_input("Podaj jaki znak wprowadziłeś")
        if true_sign:
            if true_sign == prediction:
                st.balloons()
                col2.success(f"Znak został prawidłowo sklasyfikowany! :D")
            else:
                col2.error("Niestety tym razem się nie udało :(")


elif option == "ResNet50 (PyTorch)":
    st.write("MODEL RESNET50")

    uploaded_file = st.file_uploader("Wprowadź zdjęcie")
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        col1.image(uploaded_file)

        prediction = classify_sign_resnet50_pytorch(uploaded_file)
        col2.write(f"Predicted class: {prediction}")

        true_sign = col2.text_input("Podaj jaki znak wprowadziłeś")
        if true_sign:
            if true_sign == prediction:
                st.balloons()
                col2.success(f"Znak został prawidłowo sklasyfikowany! :D")
            else:
                col2.error("Niestety tym razem się nie udało :(")

elif option == "EfficientNet (PyTorch)":
    st.write("MODEL EFFICIENTNET (PyTorch)")

    uploaded_file = st.file_uploader("Wprowadź zdjęcie")
    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        col1.image(uploaded_file)

        prediction = classify_sign_efficientnet(uploaded_file)
        col2.write(f"Predicted class: {prediction}")

        true_sign = col2.text_input("Podaj jaki znak wprowadziłeś")
        if true_sign:
            if true_sign == prediction:
                st.balloons()
                col2.success(f"Znak został prawidłowo sklasyfikowany! :D")
            else:
                col2.error("Niestety tym razem się nie udało :(")
