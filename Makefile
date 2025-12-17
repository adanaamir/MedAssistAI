# Makefile for Medical Assistant ML System

.PHONY: help build up down restart logs test clean train deploy

# Default target
help:
	@echo "Medical Assistant ML System - Docker Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  build       - Build all Docker images"
	@echo "  up          - Start all services"
	@echo "  down        - Stop all services"
	@echo "  restart     - Restart all services"
	@echo "  logs        - View logs from all services"
	@echo "  test        - Run all tests"
	@echo "  train       - Run ML training pipeline"
	@echo "  clean       - Remove containers, volumes, and images"
	@echo "  deploy      - Deploy to production"
	@echo "  status      - Check service status"
	@echo "  shell-api   - Access API container shell"
	@echo "  shell-worker - Access worker container shell"

# Build Docker images
build:
	@echo "🔨 Building Docker images..."
	docker-compose build --no-cache

# Start all services
up:
	@echo "🚀 Starting all services..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "📊 API: http://localhost:8001"
	@echo "📊 Prefect UI: http://localhost:4200"

# Stop all services
down:
	@echo "🛑 Stopping all services..."
	docker-compose down

# Restart services
restart:
	@echo "🔄 Restarting services..."
	docker-compose restart

# View logs
logs:
	docker-compose logs -f

# View API logs only
logs-api:
	docker-compose logs -f medical-api

# View worker logs only
logs-worker:
	docker-compose logs -f prefect-worker

# Run tests
test:
	@echo "🧪 Running tests..."
	docker-compose exec medical-api pytest tests/ -v

# Run ML training pipeline
train:
	@echo "🤖 Running ML training pipeline..."
	docker-compose exec prefect-worker python scripts/prefect_pipeline.py

# Check service status
status:
	@echo "📊 Service Status:"
	docker-compose ps

# Access API container shell
shell-api:
	docker-compose exec medical-api bash

# Access worker container shell
shell-worker:
	docker-compose exec prefect-worker bash

# Access database shell
shell-db:
	docker-compose exec postgres psql -U prefect -d prefect

# Clean up everything
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Cleanup complete!"

# Deploy to production
deploy:
	@echo "🚀 Deploying to production..."
	docker-compose -f docker-compose.prod.yml up -d --build
	@echo "✅ Production deployment complete!"

# Stop production
deploy-down:
	docker-compose -f docker-compose.prod.yml down

# View production logs
deploy-logs:
	docker-compose -f docker-compose.prod.yml logs -f

# Quick test API
test-api:
	@echo "🧪 Testing API endpoint..."
	curl http://localhost:8001/
	@echo ""
	@echo "🧪 Testing prediction endpoint..."
	curl -X POST http://localhost:8001/predict \
		-H "Content-Type: application/json" \
		-d '{"symptoms_text": "fever cough headache fatigue"}'

# Setup environment
setup:
	@echo "⚙️ Setting up environment..."
	cp .env.example .env
	@echo "✅ Created .env file - please edit with your values"
	@echo "📝 Run 'make up' to start services"

# Full rebuild
rebuild:
	@echo "🔨 Full rebuild..."
	docker-compose down -v
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Rebuild complete!"

# Health check
health:
	@echo "🏥 Checking service health..."
	@docker ps --format "table {{.Names}}\t{{.Status}}"

# Show resource usage
stats:
	docker stats --no-stream

# Backup models
backup-models:
	@echo "💾 Backing up models..."
	@mkdir -p backups
	@tar -czf backups/models-$(shell date +%Y%m%d-%H%M%S).tar.gz models/
	@echo "✅ Models backed up!"

# Restore models
restore-models:
	@echo "📦 Restoring models..."
	@read -p "Enter backup filename: " backup; \
	tar -xzf backups/$$backup -C .
	@echo "✅ Models restored!"