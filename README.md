# Potato Disease Classification

A web application for detecting potato leaf diseases using deep learning.

## 🚀 Live Demo

Access the app here: [Project Web UI](https://potatodiseaseclassification-fe2b.onrender.com)

---

## 📝 Features

- Upload potato leaf images and get instant disease predictions.
- Supports detection of:
  - Early Blight
  - Late Blight
  - Healthy leaves
- Simple, user-friendly web interface.
- Powered by FastAPI and TensorFlow Serving for scalable, production-grade inference.

---

## 🖼️ How to Use

1. Visit the [live site](https://potatodiseaseclassification-fe2b.onrender.com).
2. Click **Upload** and select a potato leaf image (JPG/PNG).
3. View the predicted disease class and confidence score.

---

## 🛠️ Tech Stack

- **Frontend:** HTML, JavaScript (vanilla or React)
- **Backend:** FastAPI (Python)
- **Model Serving:** TensorFlow Serving (Docker)
- **Deployment:** Render.com
- **CI/CD:** GitHub Actions

---


### Backend Setup

```bash
# Clone the repo
git clone https://github.com/Peter-Opapa/PotatoDiseaseClassification.git
cd PotatoDiseaseClassification

# Install Python dependencies
pip install -r api/requirements.txt

# Start FastAPI backend
cd src
python main.py
```

## ⚙️ Deployment

- Automated deployment via [Render.com](https://potatodiseaseclassification-fe2b.onrender.com) using GitHub Actions.
- See `.github/workflows/deploy_to_render.yaml` for CI/CD configuration.

![render success](images\render_progress.png)

---

## 📄 License

This project is licensed under the MIT License.

---
