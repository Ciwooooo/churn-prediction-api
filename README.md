# Churn Prediction API

Production-ready customer churn prediction service with PostgreSQL logging.

## Quick Start

### Single Container (Development)
\`\`\`bash
docker build -t churn-api .
docker run -p 5000:5000 churn-api
\`\`\`

### Multi-Container (Production-like)
\`\`\`bash
docker-compose up -d
\`\`\`

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /batch-predict` - Batch predictions  
- `GET /stats` - Prediction statistics

## Architecture

\`\`\`
┌─────────────┐      ┌───────────────┐      ┌─────────────┐
│   Client    │─────▶│   Flask API   │─────▶│  PostgreSQL │
└─────────────┘      └───────────────┘      └─────────────┘
                             │
                             ▼
                     ┌───────────────┐
                     │  ML Model     │
                     │ (RandomForest)│
                     └───────────────┘
\`\`\`

## Tech Stack

- **ML**: scikit-learn, pandas
- **API**: Flask, Gunicorn
- **Database**: PostgreSQL, SQLAlchemy
- **Containerization**: Docker, Docker Compose

## Future Enhancements

- [ ] Deploy to Kubernetes
- [ ] Add MLflow tracking
- [ ] Implement CI/CD pipeline
- [ ] Add Prometheus monitoring
- [ ] Create Helm charts
\`\`\`500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)