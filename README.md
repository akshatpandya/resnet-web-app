# ResNet50 Image Classification Web App

A Streamlit web app that captures images from your camera, runs ResNet50 inference, and displays the top-N predictions.

## Features

- Camera capture via browser webcam (requires HTTPS in production)
- Optional image file upload
- Configurable top-N predictions (1–10)
- Pretrained ResNet50 (ImageNet) model

## Local Development

### Prerequisites

- Python 3.9+

### Setup

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Deployment

The app can be hosted online using several platforms. Choose one:

### Option 1: Streamlit Community Cloud (recommended)

1. Push this project to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Sign in with GitHub and click "New app".
4. Select your repo, set the main file to `app.py`, and deploy.
5. Your app will be available at a URL like `https://your-app-name.streamlit.app`.

**Note:** First load may take a few minutes while PyTorch and the model download.

### Option 2: Render

1. Push this project to a GitHub repository.
2. Go to [render.com](https://render.com) and create a new Web Service.
3. Connect your repo and configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
4. Deploy. Render will assign a public URL.

### Option 3: Railway

1. Push to GitHub and connect to [railway.app](https://railway.app).
2. Create a new project from the repo.
3. Set the start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
4. Add `requirements.txt` (auto-detected). Deploy.

### Camera access in production

Camera capture requires **HTTPS**. All of the above platforms serve over HTTPS, so the browser will allow camera access once the user grants permission.
