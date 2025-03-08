# Chest X-Ray Pneumonia Detector xAi + Web app

## Project Overview

Developed a pneumonia detection system using MobileNet-V3 with transfer learning, achieving 91% accuracy in detecting pneumonia from chest X-rays. Optimized the model for edge devices through knowledge distillation to reduce model size and improve performance in resource-constrained environments.
Deployed the model on AWS SageMaker using Flask, ensuring seamless integration and scalability for production environments. Implemented Grad-CAM for model interpretability, visualizing CNN focus areas to enhance prediction accuracy and provide insight into model decision-making.

## Dataset Overview

- **Normal images count in training set:** 1349
- **Pneumonia images count in training set:** 3883
- **Total Count of images:** 5232

## Project Structure

```
Chest_Xray_Pneumonia_Detector/
│── data/
│   ├── pneumonia/
│   ├── normal/
│── train/
│   ├── model_training_code.ipynb
│   ├── saving_model/
│       ├── pneumonia_model.keras
│── results/
│       ├── sample1.png
│       ├── sample3.png
│       ├──.....
│── deployment/
    ├── app.py
    ├── requirements.txt
    ├── index.html
```

## Installation & Setup

```bash
pip install -r requirements.txt
```

## Model Training (Google Colab)

1. Install dependencies
   ```bash
   !pip install -r requirements.txt
   ```
2. Prepare dataset
   ```
   Chest_Xray_Pneumonia_Detector/
       pneumonia /
           img1.jpg
           img2.jpg
       normal/
           img1.jpg
           img2.jpg
   ```
3. Train the model using `model_training_code.ipynb`

## Deployment Guide

### 1. Local Deployment (Flask)

1. Navigate to the `deployment/` folder and run:
   ```bash
   python app.py
   ```
2. The web app will be available at `http://127.0.0.1:5000`

### 2. AWS EC2 Deployment

1. Launch an EC2 instance with Ubuntu.
2. Install dependencies:
   ```bash
   sudo apt update && sudo apt install python3-pip
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Configure security groups to allow inbound traffic on port 5000.

## API Endpoints

- `POST /predict` - Upload a chest X-ray and get a pneumonia prediction.



## Results & Interpretability
- Achieved **91% accuracy**
- Used **Grad-CAM** to visualize model focus areas for interpretability(xAI).

---

## Contact
👤 **Author:** Sai Krishna Chowdary Chundru  
📩 **Email:** [cchsaikrishnachowdary@gmail.com](mailto:cchsaikrishnachowdary@gmail.com)  
🔗 **LinkedIn:** [linkedin.com/in/sai-krishna-chowdary-chundru](https://linkedin.com/in/sai-krishna-chowdary-chundru)  
💻 **GitHub:** [github.com/sAI-2025](https://github.com/sAI-2025)  

---

## License
This project is licensed under the **MIT License** – feel free to use and modify!


