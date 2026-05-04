#!/bin/bash
# ============================================================
# SiliQun v2.0 Aziz HPC Deployment Manager
# Usage: ./deploy_aziz.sh [start|stop|status|logs|url]
# ============================================================

AZIZ_USER="ralshehri0468"
AZIZ_HOST="klogin3.aziz.hpc.kau.edu.sa"
REMOTE_DIR="~/siliqun/deploy"
PBS_SCRIPT="aziz_siliqun_service.pbs"

# SSH options
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

# ngrok tunnel (update when tunnel changes)
NGROK_HOST="${NGROK_HOST:-5.tcp.eu.ngrok.io}"
NGROK_PORT="${NGROK_PORT:-13592}"

ssh_cmd() {
    sshpass -p "r3d@29e" ssh $SSH_OPTS \
        -p "$NGROK_PORT" \
        "${AZIZ_USER}@${NGROK_HOST}" \
        "$@"
}

case "${1:-status}" in
    start)
        echo "=== Deploying SiliQun API to Aziz HPC ==="
        # Copy deploy scripts to Aziz
        sshpass -p "r3d@29e" scp $SSH_OPTS \
            -P "$NGROK_PORT" \
            deploy/aziz_siliqun_service.pbs \
            deploy/uvicorn_log_config.json \
            "${AZIZ_USER}@${NGROK_HOST}:~/siliqun/deploy/"
        
        # Submit the PBS job
        JOB_ID=$(ssh_cmd "mkdir -p ~/siliqun/deploy ~/logs && qsub ~/siliqun/deploy/aziz_siliqun_service.pbs")
        echo "Submitted job: $JOB_ID"
        echo "$JOB_ID" > .aziz_job_id
        echo "Monitor with: ./deploy_aziz.sh status"
        ;;
    
    stop)
        if [ -f .aziz_job_id ]; then
            JOB_ID=$(cat .aziz_job_id)
            ssh_cmd "qdel $JOB_ID 2>/dev/null && echo 'Job $JOB_ID cancelled' || echo 'Job not found'"
            rm -f .aziz_job_id
        else
            echo "No job ID found. Check manually with: ./deploy_aziz.sh status"
        fi
        ;;
    
    status)
        echo "=== SiliQun API Job Status ==="
        ssh_cmd "qstat -u $AZIZ_USER 2>/dev/null | grep -E 'siliqun|Job ID' || echo 'No SiliQun jobs running'"
        echo ""
        echo "=== Service URL ==="
        ssh_cmd "cat ~/siliqun_service_url.txt 2>/dev/null || echo 'Service not started yet'"
        ;;
    
    logs)
        echo "=== SiliQun API Logs (last 50 lines) ==="
        ssh_cmd "ls -t ~/logs/siliqun_api_*.log 2>/dev/null | head -1 | xargs tail -50 2>/dev/null || echo 'No logs found yet'"
        ;;
    
    url)
        ssh_cmd "cat ~/siliqun_service_url.txt 2>/dev/null || echo 'Service not started'"
        ;;
    
    test)
        SERVICE_URL=$(ssh_cmd "cat ~/siliqun_service_url.txt 2>/dev/null | head -1")
        if [ -z "$SERVICE_URL" ]; then
            echo "Service URL not found. Is the service running?"
            exit 1
        fi
        echo "Testing SiliQun API at $SERVICE_URL"
        echo ""
        echo "--- Health Check ---"
        curl -s "${SERVICE_URL}/health" | python3 -m json.tool
        echo ""
        echo "--- Device Profiles ---"
        curl -s "${SERVICE_URL}/devices" | python3 -m json.tool
        echo ""
        echo "--- Circuit Simulation (Bell State) ---"
        curl -s -X POST "${SERVICE_URL}/simulate/circuit" \
            -H "Content-Type: application/json" \
            -d '{
                "qasm": "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; h q[0]; cx q[0],q[1];",
                "device": "simos",
                "shots": 1024
            }' | python3 -m json.tool
        ;;
    
    *)
        echo "Usage: $0 [start|stop|status|logs|url|test]"
        exit 1
        ;;
esac
