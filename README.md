# 🌾 Crop Prediction Dashboard

Welcome to the Crop Prediction Dashboard, a web application that uses Machine Learning to recommend the most suitable crops based on climatic and soil conditions. Built using a Multi-Layer Perceptron (MLP) deep learning model, this project features an interactive interface using Streamlit.

![Dashboard Screenshot](path_to_screenshot.png)

## 🚀 Features

- **Crop Prediction**: Predict the optimal crop to plant based on input features like temperature, humidity, rainfall, and more.
- **Visualizations**: Explore various visualizations to understand the data and predictions better.
- **Interactive Dashboard**: A user-friendly interface built with Streamlit.
- **Model Integration**: Utilizes an MLP model for crop recommendation.

## 📂 Project Structure

```
Crop-Recommendation/
├── StreamLit/
│   ├── Index.py              # Main Streamlit entry point
│   └── Pages/
│       ├── visualization.py  # Visualization page
│       └── prediction.py     # Prediction page
├── Notebook/
│   ├── dataset/              # Directory containing datasets
│   ├── Aggericulture_Neural_Network.ipynb   # Jupyter notebook for data analysis and model training
│   ├── crop_model.h5         # Pre-trained MLP model
│   └── scaler.pkl            # Scaler for feature normalization
└── README.md                 # Project README
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- Pip (Python package installer)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kendor74/Crop-Recommendation.git
   cd Crop-Recommendation
   ```

2. **Install the required packages:**

   ```bash
   pip install -r sreamlit
   ```

3. **Run the Streamlit application:**

   ```bash
   streamlit run StreamLit/Index.py
   ```

4. **Access the dashboard:**

   Open your web browser and navigate to `http://localhost:8501`.

## 🧠 Machine Learning Model

The model uses an MLP (Multi-Layer Perceptron) deep learning architecture, implemented with TensorFlow/Keras. The model predicts the best crop for given climatic and soil conditions.

### Model Training

The model was trained and evaluated in the [Aggericulture_Neural_Network.ipynb](Notebook/Aggericulture_Neural_Network.ipynb) notebook. Key steps include:

- **Data Preprocessing**: Cleaning and preparing the data.
- **Feature Engineering**: Creating features like SFI, SMD, THI, and SQI.
- **Modeling**: Building and training the MLP model.
- **Evaluation**: Assessing the model's performance.

## 📊 Visualizations

Explore various visualizations to gain insights into the data:

- **Heatmaps**: Average climatic and soil requirements by crop.
- **Histograms**: Distribution of temperature, humidity, rainfall, and SQI.
- **Pair Plots**: Relationships between temperature, humidity, rainfall, and SQI.
- **Scatter Plots**: Crop recommendations based on SQI and THI.
- **Scatter Matrix**: Multi-dimensional feature relationships.

## 🌐 Deployment

### Deploying Locally

To run the Streamlit application locally:

```bash
streamlit run StreamLit/Index.py
```

## 🛠️ Technologies Used

- **Python**: The core language for this project.
- **Streamlit**: For building the interactive web application.
- **TensorFlow/Keras**: For the MLP deep learning model.
- **Plotly & Seaborn**: For data visualization.

## 🤝 Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.


