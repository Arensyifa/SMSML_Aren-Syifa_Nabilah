from fastapi import FastAPI, Response
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
import psutil
import threading
import time
import json

app = FastAPI()

REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests")
PRED_COUNT = Counter("model_prediction_count_total", "Total model predictions")
ERROR_COUNT = Counter("model_error_count_total", "Total prediction errors")
INFERENCE_TIME = Histogram("model_inference_time_seconds", "Model inference time")

CPU_USAGE = Gauge("system_cpu_usage", "CPU Usage Percentage")
RAM_USAGE = Gauge("system_ram_usage", "RAM Usage Percentage")
DISK_USAGE = Gauge("system_disk_usage", "Disk Usage Percentage")

def metrics_loop():
    while True:
        CPU_USAGE.set(psutil.cpu_percent())
        RAM_USAGE.set(psutil.virtual_memory().percent)
        DISK_USAGE.set(psutil.disk_usage("/").percent)
        time.sleep(2)

@app.on_event("startup")
def startup_event():
    threading.Thread(target=metrics_loop, daemon=True).start()

@app.post("/update")
def update_metrics(payload: dict):
    REQUEST_COUNT.inc()
    PRED_COUNT.inc()
    INFERENCE_TIME.observe(payload["latency"])
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
