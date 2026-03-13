# AI Patient Readmission Prediction System

See full documentation in [`docs/README.md`](docs/README.md).

## Quick Start

```bash
# 1. Train the model
cd ml && pip install -r requirements.txt
python train_model.py

# 2. Start the API
cd ../backend && pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 3. Start the UI
cd ../frontend && npm install && npm start
```

Open http://localhost:3000
