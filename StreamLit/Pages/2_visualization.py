import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

st.title("Crop Data Visualizations")

# Sample data (Replace with actual data or load it as needed)
@st.cache
def load_data():
    # Load your dataset here
    # Example placeholder data
    data = pd.read_csv("C:\\Users\\pc\\Downloads\\crop_data.csv")
    return data

df = load_data()

# Page title
st.title("Modern Crop Recommendation Visualizations")

# Sidebar for selecting visualization
visualization_type = st.sidebar.radio("Choose Visualization", [
    "Average Climatic and Soil Requirements (Heatmap)",
    "Feature Distributions",
    "Pair Plot",
    "SQI vs THI Scatter Plot",
    "Feature Distributions with KDE",
    "Scatter Matrix",
    "Correlation Heatmap"
])

if visualization_type == "Average Climatic and Soil Requirements (Heatmap)":
    # Group by label and calculate the mean
    features = ['temperature', 'humidity', 'rainfall', 'sqi']
    average_requirements = df.groupby('label')[features].mean().reset_index()

    heatmap_data = average_requirements.set_index('label')
    fig = ff.create_annotated_heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns.tolist(),
        y=heatmap_data.index.tolist(),
        colorscale='Viridis',
        annotation_text=np.round(heatmap_data.values, 2),
        showscale=True
    )
    fig.update_layout(title="Average Climatic and Soil Requirements by Crop")
    st.plotly_chart(fig)

elif visualization_type == "Feature Distributions":
    fig = plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    sns.histplot(df['temperature'], kde=True, color='skyblue')
    plt.title('Temperature Distribution')

    plt.subplot(2, 2, 2)
    sns.histplot(df['humidity'], kde=True, color='olive')
    plt.title('Humidity Distribution')

    plt.subplot(2, 2, 3)
    sns.histplot(df['rainfall'], kde=True, color='gold')
    plt.title('Rainfall Distribution')

    plt.subplot(2, 2, 4)
    sns.histplot(df['sqi'], kde=True, color='teal')
    plt.title('SQI Distribution')

    plt.tight_layout()
    st.pyplot(fig)

elif visualization_type == "Pair Plot":

    # Assuming df is your DataFrame
    fig = px.scatter_matrix(df,
                            dimensions=['temperature', 'humidity', 'rainfall', 'sqi'],
                            color='label',
                            title='Pair Plot of Temperature, Humidity, Rainfall, and SQI',
                            labels={'label': 'Crop Type'},
                            color_continuous_scale=px.colors.sequential.Viridis)  # Changed 'Husl' to 'Viridis'

    # Update layout for better visualization
    fig.update_layout(height=800, width=800, title_x=0.5)

    # Show plot in Streamlit
    st.plotly_chart(fig)



elif visualization_type == "SQI vs THI Scatter Plot":
    fig = px.scatter(df, x='sqi', y='thi', color='label', size='CWR', hover_data=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    fig.update_layout(title='Crop Recommendation based on SQI and THI', xaxis_title='SQI', yaxis_title='THI')
    st.plotly_chart(fig)

elif visualization_type == "Feature Distributions with KDE":
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'Kc', 'sfi', 'pet', 'smd', 'thi', 'sqi', 'CWR']
    for feature in features:
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        st.pyplot(fig)

elif visualization_type == "Scatter Matrix":
    scatter_matrix = px.scatter_matrix(df, dimensions=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], color='label')
    scatter_matrix.update_layout(title='Scatter Matrix of Features', width=1200, height=800)
    st.plotly_chart(scatter_matrix)

elif visualization_type == "Correlation Heatmap":
    t = df.copy()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    t['label'] = le.fit_transform(t['label'])

    plt.figure(figsize=(15, 8))
    sns.heatmap(t.corr(), annot=True, cmap='coolwarm', linewidths=.5)
    plt.title("Correlation Heatmap")
    st.pyplot()