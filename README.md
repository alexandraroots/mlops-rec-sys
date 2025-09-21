# mlops-reс-sys

```bash
docker build -t mlops-rec-sys -f docker/Dockerfile .
```

```bash
docker run -d --name mlops-server -p 50051:50051 mlops-rec-sys
```
