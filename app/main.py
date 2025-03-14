import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
    column_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    data = pd.read_csv('data/data.csv')
    data = data.iloc[:, 2:]
    data.columns = column_names
    return data

def get_scaled_data(input_data):
    data = get_clean_data()
    
    scaled_dict = {}
    for key,value in input_data.items():
        max_value = data[key].max()
        min_value = data[key].min()
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_dict[key] = scaled_value
    return scaled_dict
        
def get_radar_chart(input_data):
    
    input_data=get_scaled_data(input_data)
    
    categories=['Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity','Concave points','Symmetry','Fractal dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
           input_data["radius_mean"],input_data["texture_mean"],input_data["perimeter_mean"],input_data["area_mean"],input_data["smoothness_mean"],
           input_data["compactness_mean"],input_data["concavity_mean"],input_data["concave points_mean"],input_data["symmetry_mean"],input_data["fractal_dimension_mean"] 
           ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data["radius_se"],input_data["texture_se"],input_data["perimeter_se"],input_data["area_se"],input_data["smoothness_se"],
           input_data["compactness_se"],input_data["concavity_se"],input_data["concave points_se"],input_data["symmetry_se"],input_data["fractal_dimension_se"] 
           ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[input_data["radius_worst"],input_data["texture_worst"],input_data["perimeter_worst"],input_data["area_worst"],input_data["smoothness_worst"],
           input_data["compactness_worst"],input_data["concavity_worst"],input_data["concave points_worst"],input_data["symmetry_worst"],input_data["fractal_dimension_worst"] 
           ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig
    
# Add the sidebar
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data=get_clean_data()
    
    # Define the labels
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    # Add the sliders
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )  
    
    return input_dict

def add_predictions(input_data):
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    
    input_array=np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction=model.predict(input_array_scaled)
    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is predicted to be: ")
    if prediction[0]==0:
        st.markdown("<p style='color:blue; font-size:28px;'>Benign</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:red; font-size:28px;'>Malignant</p>", unsafe_allow_html=True)
    
    
    st.write("Probability of being Benign: ",model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being Malignant: ",model.predict_proba(input_array_scaled)[0][1])
    st.markdown("<p style='color:#b85e5e; font-size:15px;'>This app can assist medical professionals in diagnosing breast cancer. It is not a substitute for professional medical advice, diagnosis or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>", unsafe_allow_html=True)
    

def main():

    st.set_page_config(page_title="Breast Cancer Diagnosis",
                    page_icon="👩‍⚕️", 
                    layout="wide", 
                    initial_sidebar_state="expanded")

    # Add the sidebar
    input_data = add_sidebar()
    
 
    # Add the structure
    with st.container():
        st.title("Breast Cancer Diagnosis")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
    col1, empty_col, col2 = st.columns([4, 0.5, 1.5])  # Adjust middle column for spacing

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        
        add_predictions(input_data)




if __name__ == '__main__':
    main()
