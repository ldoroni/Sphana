---
title: Deployment
description: Docker, Kubernetes, and production deployment
---

# Deployment

This guide covers deploying Sphana Database in production environments using Docker and Kubernetes.

## Docker Deployment

### Basic Docker Setup

**Dockerfile** (included in `services/Sphana.Database/`):

```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS base
WORKDIR /app
EXPOSE 5000 5001

# Install CUDA runtime (for GPU)
RUN apt-get update && apt-get install -y \
    cuda-runtime-12-8 \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY ["Sphana.Database/Sphana.Database.csproj", "Sphana.Database/"]
COPY ["Sphana.Database.Protos/Sphana.Database.Protos.csproj", "Sphana.Database.Protos/"]
RUN dotnet restore "Sphana.Database/Sphana.Database.csproj"

COPY . .
WORKDIR "/src/Sphana.Database"
RUN dotnet build "Sphana.Database.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "Sphana.Database.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
COPY models/ /app/models/

ENTRYPOINT ["dotnet", "Sphana.Database.dll"]
```

### Build and Run

```bash
cd services/Sphana.Database

# Build image
docker build -t sphana-database:latest .

# Run with GPU
docker run --gpus all -p 5000:5000 -p 5001:5001 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    sphana-database:latest

# Run without GPU (CPU only)
docker run -p 5000:5000 -p 5001:5001 \
    -e Sphana__Models__UseGpu=false \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    sphana-database:latest
```

### Docker Compose

**docker-compose.yml**:

```yaml
services:
  sphana-database:
    build: .
    image: sphana-database:latest
    ports:
      - "5000:5000"
      - "5001:5001"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - Sphana__Models__UseGpu=true
      - Sphana__VectorIndex__StoragePath=/app/data/vector_index
      - Sphana__KnowledgeGraph__GraphStoragePath=/app/data/knowledge_graph
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**docker-compose.gpu.yml** (GPU-specific):

```yaml
services:
  sphana-database:
    extends:
      file: docker-compose.yml
      service: sphana-database
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_VISIBLE_DEVICES=0
```

Start services:

```bash
# CPU
docker-compose up -d

# GPU
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## Kubernetes Deployment

### Basic Deployment

**deployment.yaml**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sphana-database
  namespace: sphana
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sphana-database
  template:
    metadata:
      labels:
        app: sphana-database
    spec:
      containers:
      - name: sphana-database
        image: sphana-database:latest
        ports:
        - containerPort: 5000
          name: http
        - containerPort: 5001
          name: grpc
        env:
        - name: Sphana__Models__UseGpu
          value: "true"
        - name: ASPNETCORE_URLS
          value: "http://+:5000;https://+:5001"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: models
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: sphana-data-pvc
      - name: models
        persistentVolumeClaim:
          claimName: sphana-models-pvc
```

### Service

**service.yaml**:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: sphana-database
  namespace: sphana
spec:
  selector:
    app: sphana-database
  ports:
  - name: http
    port: 5000
    targetPort: 5000
  - name: grpc
    port: 5001
    targetPort: 5001
  type: LoadBalancer
```

### Persistent Volumes

**pvc.yaml**:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: sphana-data-pvc
  namespace: sphana
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: sphana-models-pvc
  namespace: sphana
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
```

### ConfigMap

**configmap.yaml**:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sphana-config
  namespace: sphana
data:
  appsettings.Production.json: |
    {
      "Sphana": {
        "Models": {
          "UseGpu": true,
          "EmbeddingModelPath": "/app/models/embedding.onnx",
          "RelationExtractionModelPath": "/app/models/relation_extraction.onnx",
          "GnnRankerModelPath": "/app/models/gnn_ranker.onnx"
        },
        "VectorIndex": {
          "StoragePath": "/app/data/vector_index"
        },
        "KnowledgeGraph": {
          "GraphStoragePath": "/app/data/knowledge_graph"
        }
      }
    }
```

### Apply Configuration

```bash
kubectl create namespace sphana
kubectl apply -f pvc.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### GPU Node Pool

For GPU nodes, use node selectors:

```yaml
spec:
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

## Horizontal Scaling

### Stateless Query Frontends

Scale query replicas independently:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sphana-database-hpa
  namespace: sphana
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sphana-database
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Sharding Strategy

For large-scale deployments:

1. **Shard by Tenant**: Each tenant on separate deployment
2. **Shard by Index**: Large indexes split across nodes
3. **Read Replicas**: Separate ingestion and query clusters

## Monitoring

### Prometheus

**servicemonitor.yaml**:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sphana-database
  namespace: sphana
spec:
  selector:
    matchLabels:
      app: sphana-database
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
```

### Grafana Dashboard

Import dashboard template from `monitoring/grafana-dashboard.json`.

Key metrics:
- Query latency (p50, p95, p99)
- Throughput (qps)
- GPU utilization
- Memory usage
- Index size

## Production Checklist

### Performance

- [ ] Use INT8 quantized models
- [ ] Enable GPU acceleration
- [ ] Configure appropriate batch sizes
- [ ] Use NVMe SSDs for storage
- [ ] Enable HTTP/2 for gRPC

### Reliability

- [ ] Configure health checks
- [ ] Set resource limits
- [ ] Enable persistent storage
- [ ] Configure backup strategy
- [ ] Set up monitoring and alerts

### Security

- [ ] Enable TLS for gRPC
- [ ] Configure network policies
- [ ] Use secrets for sensitive config
- [ ] Enable audit logging
- [ ] Restrict container privileges

### Scalability

- [ ] Configure HPA
- [ ] Use load balancer
- [ ] Plan sharding strategy
- [ ] Optimize batch processing
- [ ] Monitor resource usage

## Cost Optimization

### GPU Selection

| GPU | Memory | Price/hr | Throughput | Cost/1M queries |
|-----|--------|----------|------------|-----------------|
| T4 | 16GB | $0.35 | 1000 qps | $0.10 |
| A10 | 24GB | $0.73 | 2000 qps | $0.10 |
| A100 | 40GB | $3.67 | 5000 qps | $0.20 |

### Instance Types

- **Query-only**: 4 vCPU, 16GB RAM, 1× T4 GPU
- **Ingestion-only**: 8 vCPU, 32GB RAM, no GPU
- **All-in-one**: 8 vCPU, 32GB RAM, 1× T4 GPU

## Next Steps

- Review [Database Architecture](/database-arch)
- Learn about [Integration](/integration) patterns
- Explore [Advanced Topics](/advanced) for optimization

