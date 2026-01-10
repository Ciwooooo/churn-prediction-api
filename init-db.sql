-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_prediction_timestamp 
ON prediction_logs(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_churn 
ON prediction_logs(churn_prediction);

-- Create a view for daily statistics
CREATE OR REPLACE VIEW daily_prediction_stats AS
SELECT 
    DATE(timestamp) as prediction_date,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN churn_prediction THEN 1 ELSE 0 END) as churn_predictions,
    AVG(churn_probability) as avg_churn_probability,
    AVG(response_time_ms) as avg_response_time
FROM prediction_logs
GROUP BY DATE(timestamp)
ORDER BY prediction_date DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE churn_db TO churn_user;