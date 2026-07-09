# 🧬 TwinCare AI — Personal Health Intelligence Platform

> **AI-powered Digital Twin** that transforms blood reports into actionable health intelligence with explainable risk prediction and a conversational health copilot.

![Status](https://img.shields.io/badge/status-hackathon%20MVP-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c)
![AMD ROCm](https://img.shields.io/badge/AMD-ROCm%20compatible-ed1c24)

---

## 🚀 What It Does

Upload a blood report → AI extracts biomarkers → Digital Twin updates → Heart disease risk predicted with SHAP explanations → Ask your AI Health Copilot about your data.

**One Demo Flow:**
1. **Sign up** with email/password
2. **Upload** a blood report (PDF/image)
3. **See** extracted biomarkers and health score on the Dashboard
4. **View** heart disease risk prediction with explainable AI (SHAP waterfall)
5. **Ask** the AI Copilot: "What does my cholesterol level mean?"

---

## 🏗️ Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   React Frontend │────▶│  FastAPI Backend  │────▶│   PostgreSQL     │
│   (TypeScript)   │     │  (Python 3.11)   │     │   (Data Store)   │
└──────────────────┘     └────────┬─────────┘     └──────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
            ┌──────────┐  ┌──────────┐  ┌──────────┐
            │ OCR      │  │ PyTorch  │  │ Fireworks│
            │ Pipeline │  │ Heart    │  │ API      │
            │(Tesseract)│  │ Model   │  │ (Gemma)  │
            └──────────┘  │ + SHAP   │  └──────────┘
                          │ (ROCm)   │
                          └──────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React + TypeScript + TailwindCSS | UI via Natively Builder |
| Backend | FastAPI + SQLAlchemy + PostgreSQL | REST API + business logic |
| OCR | Tesseract + pdf2image | Report text extraction |
| Risk Model | PyTorch MLP (ROCm) | Heart disease prediction |
| Explainability | SHAP | Feature attribution |
| LLM | Fireworks API (Gemma 2) | Health Copilot + NER fallback |
| Deployment | Docker Compose | One-command launch |

---

## ⚡ Quick Start

### Prerequisites
- Docker & Docker Compose
- (Optional) Fireworks AI API key for the Copilot

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/twincareAI.git
cd twincareAI
cp .env.example .env
# Edit .env and add your FIREWORKS_API_KEY
```

### 2. Launch

```bash
docker compose up --build
```

### 3. Seed Demo Data (Optional)

```bash
docker compose exec backend python seed_data/seed.py
```

### 4. Access

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

**Demo Login:** `demo@twincare.ai` / `demo123`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | Login, get JWT token |
| GET | `/api/v1/auth/me` | Get current user profile |
| POST | `/api/v1/reports/upload` | Upload health report |
| GET | `/api/v1/reports/{id}/status` | Check processing status |
| GET | `/api/v1/reports` | List all reports |
| GET | `/api/v1/reports/{id}/biomarkers` | Get extracted biomarkers |
| GET | `/api/v1/digital-twin` | Get Digital Twin state |
| GET | `/api/v1/risk-predictions` | Get risk predictions |
| GET | `/api/v1/risk-predictions/{id}` | Get prediction details + SHAP |
| POST | `/api/v1/copilot/chat` | Chat with AI Health Copilot |
| GET | `/api/v1/copilot/history` | Get chat history |
| GET | `/api/v1/dashboard` | Get aggregated dashboard data |

---

## 🤖 AMD GPU Usage

The heart disease prediction model is **trained and served using PyTorch on AMD ROCm**:

- **Training**: AMD GPU Jupyter Notebook (`notebooks/01_heart_disease_model_training.ipynb`)
- **Inference**: PyTorch MLP running on ROCm-compatible hardware
- **Proof**: `rocm-smi` shows GPU utilization during training and inference

The Copilot uses **Gemma 2 via Fireworks AI** (eligible for AMD Gemma bonus).

---

## 📁 Project Structure

```
twincareAI/
├── docker-compose.yml          # One-command launch
├── .env.example                # Configuration template
├── README.md                   # You are here
├── backend/
│   ├── Dockerfile
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Settings
│   ├── database.py             # PostgreSQL connection
│   ├── models/                 # SQLAlchemy models (6 tables)
│   ├── schemas/                # Pydantic request/response
│   ├── routers/                # API endpoints
│   ├── services/               # Business logic
│   ├── ai/                     # AI pipelines (OCR, model, SHAP, LLM)
│   ├── utils/                  # Security, biomarker ranges
│   └── seed_data/              # Demo data seeder
├── frontend/                   # React app (Natively Builder)
└── notebooks/                  # AMD GPU training notebooks
```

---

## ⚠️ Disclaimer

**TwinCare AI is a decision-support and preventive-health tool, NOT a diagnostic tool.** All predictions and AI responses include disclaimers. This application uses synthetic/public sample data and has not undergone clinical validation. Always consult a qualified healthcare professional for medical advice.

---

## 🔮 Future Roadmap

- Multi-modal ingestion (X-rays, CT scans, wearable data)
- 5-year health trajectory simulation
- Medical literature RAG (PubMed/clinical guidelines)
- Clinical EHR integration (HL7 FHIR)
- On-premise AMD GPU deployment for hospitals
- Multi-language Copilot support

---

## 📝 License

MIT License — Built for AMD Developer Hackathon: ACT II

---

*Built with ❤️ and AMD ROCm*
