-- Initialize database for fraud monitoring system
-- This script runs when PostgreSQL container starts

-- Create database if it doesn't exist (handled by POSTGRES_DB env var)
-- CREATE DATABASE fraud_db;

-- Connect to the fraud_db database
\c fraud_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Grant privileges to user
GRANT ALL PRIVILEGES ON DATABASE fraud_db TO fraud_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO fraud_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fraud_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fraud_user;

-- Create indexes for better performance (these will be created by Alembic migrations)
-- But we can add some additional performance optimizations

-- Log the initialization
\echo 'Database initialization completed successfully'