-- Kalshi Trading Bot - PostgreSQL Initialization Script
-- This script is automatically run when the PostgreSQL container starts for the first time

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Set timezone
SET timezone = 'UTC';

-- Create database if it doesn't exist (already created by POSTGRES_DB env var)
-- But we can set some database-level settings

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE kalshi_trading TO kalshi;

-- Create schema if needed
CREATE SCHEMA IF NOT EXISTS trading AUTHORIZATION kalshi;

-- Set search path
ALTER DATABASE kalshi_trading SET search_path TO public, trading;

-- Create indexes for common queries (tables will be created by application)
-- These will be created when the tables exist

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Kalshi Trading Database initialized successfully';
    RAISE NOTICE 'Database: kalshi_trading';
    RAISE NOTICE 'User: kalshi';
    RAISE NOTICE 'Extensions: uuid-ossp, pg_stat_statements';
    RAISE NOTICE 'Timezone: UTC';
END $$;
