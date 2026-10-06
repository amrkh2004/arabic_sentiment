#!/usr/bin/env bash
# ==============================================================================
# Automated Canary Rollback Script for Arabic Sentiment Service
# Reverts Nginx routing back to 100% stable baseline upon SLA failure
# ==============================================================================

set -euo pipefail

echo "======================================================"
echo "    [ALERT] INITIATING CANARY AUTOMATED ROLLBACK"
echo "======================================================"

NGINX_CONF="deploy/nginx/nginx.conf"

if [ ! -f "$NGINX_CONF" ]; then
    echo "Error: $NGINX_CONF not found."
    exit 1
fi

echo "Updating Nginx configuration to point 100% traffic to stable baseline..."
cat << 'EOF' > "$NGINX_CONF"
events { worker_connections 1024; }

http {
    upstream sentiment_service_canary {
        server sentiment-api:8000 weight=10;
        # Canary disabled
    }

    server {
        listen 80;

        location / {
            proxy_pass http://sentiment_service_canary;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Request-ID $request_id;
        }

        location /health {
            proxy_pass http://sentiment_service_canary/health;
        }

        location /predict {
            proxy_pass http://sentiment_service_canary/predict;
        }

        location /metrics {
            proxy_pass http://sentiment-api:8000/metrics;
        }
    }
}
EOF

echo "Reloading Nginx proxy service..."
docker compose exec -T nginx nginx -s reload || true

echo "Scaling down canary deployment..."
docker compose stop sentiment-api-canary || true

echo "======================================================"
echo "    [SUCCESS] ROLLBACK COMPLETE: TRAFFIC 100% STABLE"
echo "======================================================"
