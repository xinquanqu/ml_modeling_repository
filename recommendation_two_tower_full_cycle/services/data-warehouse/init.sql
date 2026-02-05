-- =============================================================================
-- ML Platform Database Initialization
-- =============================================================================

-- Create additional databases
CREATE DATABASE mlflow;

-- =============================================================================
-- Raw Data Tables
-- =============================================================================

CREATE TABLE IF NOT EXISTS raw_data (
    id BIGSERIAL PRIMARY KEY,
    batch_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(128) NOT NULL,
    item_id VARCHAR(128) NOT NULL,
    features JSONB DEFAULT '{}',
    label FLOAT,
    source VARCHAR(64) DEFAULT 'api',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for common queries
    CONSTRAINT raw_data_batch_idx UNIQUE (batch_id, user_id, item_id)
);

CREATE INDEX idx_raw_data_user_id ON raw_data(user_id);
CREATE INDEX idx_raw_data_item_id ON raw_data(item_id);
CREATE INDEX idx_raw_data_created_at ON raw_data(created_at);
CREATE INDEX idx_raw_data_features ON raw_data USING GIN(features);

-- =============================================================================
-- Processed Data Tables
-- =============================================================================

CREATE TABLE IF NOT EXISTS processed_data (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    item_id VARCHAR(128) NOT NULL,
    
    -- Processed features
    user_features JSONB DEFAULT '{}',
    item_features JSONB DEFAULT '{}',
    interaction_features JSONB DEFAULT '{}',
    
    label FLOAT,
    split VARCHAR(16) DEFAULT 'train',  -- train, validation, test
    
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT processed_data_unique UNIQUE (user_id, item_id)
);

CREATE INDEX idx_processed_data_split ON processed_data(split);
CREATE INDEX idx_processed_data_user_id ON processed_data(user_id);
CREATE INDEX idx_processed_data_item_id ON processed_data(item_id);

-- =============================================================================
-- Feature Store Tables
-- =============================================================================

-- Feature Registry: Metadata about available features
CREATE TABLE IF NOT EXISTS feature_registry (
    id SERIAL PRIMARY KEY,
    name VARCHAR(128) NOT NULL UNIQUE,
    entity_type VARCHAR(32) NOT NULL,  -- 'user' or 'item'
    dtype VARCHAR(32) NOT NULL DEFAULT 'float32',
    description TEXT,
    version VARCHAR(16) DEFAULT '1.0',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User Features: Precomputed user embeddings/features
CREATE TABLE IF NOT EXISTS user_features (
    user_id VARCHAR(128) PRIMARY KEY,
    features JSONB NOT NULL DEFAULT '{}',
    embedding FLOAT[] DEFAULT NULL,
    version VARCHAR(16) DEFAULT '1.0',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Item Features: Precomputed item embeddings/features
CREATE TABLE IF NOT EXISTS item_features (
    item_id VARCHAR(128) PRIMARY KEY,
    features JSONB NOT NULL DEFAULT '{}',
    embedding FLOAT[] DEFAULT NULL,
    version VARCHAR(16) DEFAULT '1.0',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- Model Registry Tables
-- =============================================================================

CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(128) NOT NULL UNIQUE,
    model_type VARCHAR(64) NOT NULL DEFAULT 'two_tower',
    version VARCHAR(32) NOT NULL,
    
    -- MLflow integration
    mlflow_run_id VARCHAR(64),
    mlflow_model_uri TEXT,
    
    -- Model metadata
    metrics JSONB DEFAULT '{}',
    config JSONB DEFAULT '{}',
    
    -- Status
    is_active BOOLEAN DEFAULT false,
    is_production BOOLEAN DEFAULT false,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deployed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_model_registry_active ON model_registry(is_active);
CREATE INDEX idx_model_registry_production ON model_registry(is_production);

-- =============================================================================
-- Training Jobs Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS training_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(64) NOT NULL UNIQUE,
    
    -- Configuration
    config JSONB NOT NULL DEFAULT '{}',
    
    -- Status
    status VARCHAR(32) DEFAULT 'pending',
    current_epoch INT DEFAULT 0,
    train_loss FLOAT,
    val_loss FLOAT,
    metrics JSONB DEFAULT '{}',
    
    -- MLflow
    mlflow_run_id VARCHAR(64),
    model_uri TEXT,
    
    -- Error handling
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_jobs_created_at ON training_jobs(created_at DESC);

-- =============================================================================
-- Inference Logs Table (for observability)
-- =============================================================================

CREATE TABLE IF NOT EXISTS inference_logs (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(128) NOT NULL,
    model_version VARCHAR(32),
    
    -- Request/Response
    request JSONB DEFAULT '{}',
    response JSONB DEFAULT '{}',
    
    -- Metrics
    latency_ms FLOAT,
    num_predictions INT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_inference_logs_created_at ON inference_logs(created_at DESC);
CREATE INDEX idx_inference_logs_user_id ON inference_logs(user_id);

-- =============================================================================
-- Functions & Triggers
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_feature_registry_updated_at
    BEFORE UPDATE ON feature_registry
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_features_updated_at
    BEFORE UPDATE ON user_features
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_item_features_updated_at
    BEFORE UPDATE ON item_features
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
