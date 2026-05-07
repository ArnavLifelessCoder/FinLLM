"""Deploy FinLLM Studio as a standalone application.

This script packages the FinLLM Studio web app for deployment:
- Creates a production build
- Bundles all assets
- Generates deployment instructions
- Optionally creates a Docker container
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = PROJECT_ROOT / "webapp"
DIST_DIR = PROJECT_ROOT / "dist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy FinLLM Studio application")
    parser.add_argument("--output-dir", default="dist", help="Output directory for deployment bundle")
    parser.add_argument("--docker", action="store_true", help="Create Docker container")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    return parser.parse_args()


def create_deployment_bundle(output_dir: Path) -> None:
    """Create a deployment bundle with all necessary files."""
    print(f"Creating deployment bundle in {output_dir}")
    
    # Clean and create output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # Copy webapp files
    webapp_dist = output_dir / "webapp"
    shutil.copytree(WEBAPP_DIR, webapp_dist)
    
    # Copy source code
    src_dist = output_dir / "src"
    shutil.copytree(PROJECT_ROOT / "src", src_dist)
    
    # Copy essential files
    essential_files = [
        "requirements.txt",
        "pyproject.toml",
        "README.md",
        "MODEL_CARD.md",
        "finance_tokenizer.model",
        "finance_tokenizer.vocab",
    ]
    
    for filename in essential_files:
        src = PROJECT_ROOT / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
    
    # Copy configs
    configs_dist = output_dir / "configs"
    shutil.copytree(PROJECT_ROOT / "configs", configs_dist)
    
    # Copy data directory structure (without large files)
    data_dist = output_dir / "data"
    data_dist.mkdir()
    
    # Copy retrieval index if it exists
    retrieval_src = PROJECT_ROOT / "data" / "retrieval"
    if retrieval_src.exists():
        retrieval_dist = data_dist / "retrieval"
        shutil.copytree(retrieval_src, retrieval_dist)
    
    # Copy runs directory (checkpoints)
    runs_src = PROJECT_ROOT / "runs"
    if runs_src.exists():
        runs_dist = output_dir / "runs"
        shutil.copytree(runs_src, runs_dist)
    
    print(f"✓ Deployment bundle created in {output_dir}")


def create_dockerfile(output_dir: Path, host: str, port: int) -> None:
    """Create a Dockerfile for containerized deployment."""
    dockerfile_content = f"""FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install package
RUN pip install -e .

# Expose port
EXPOSE {port}

# Run application
CMD ["python", "webapp/server.py", "--host", "{host}", "--port", "{port}"]
"""
    
    dockerfile_path = output_dir / "Dockerfile"
    dockerfile_path.write_text(dockerfile_content)
    print(f"✓ Dockerfile created at {dockerfile_path}")
    
    # Create .dockerignore
    dockerignore_content = """__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info
dist
build
.git
.gitignore
.vscode
.idea
*.log
.DS_Store
"""
    
    dockerignore_path = output_dir / ".dockerignore"
    dockerignore_path.write_text(dockerignore_content)
    print(f"✓ .dockerignore created at {dockerignore_path}")


def create_deployment_docs(output_dir: Path, host: str, port: int) -> None:
    """Create deployment documentation."""
    docs_content = f"""# FinLLM Studio Deployment Guide

This directory contains a production-ready deployment of FinLLM Studio.

## Quick Start

### Local Deployment

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

2. Run the application:
```bash
python webapp/server.py --host {host} --port {port}
```

3. Open your browser to:
```
http://localhost:{port}
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t finllm-studio .
```

2. Run the container:
```bash
docker run -p {port}:{port} finllm-studio
```

3. Access the application at:
```
http://localhost:{port}
```

## Production Deployment

### Using a Reverse Proxy (Nginx)

```nginx
server {{
    listen 80;
    server_name your-domain.com;

    location / {{
        proxy_pass http://127.0.0.1:{port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
```

### Using Gunicorn (Production WSGI Server)

For production, consider using Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b {host}:{port} webapp.server:app
```

### Environment Variables

- `FINLLM_HOST`: Host to bind to (default: {host})
- `FINLLM_PORT`: Port to bind to (default: {port})
- `FINLLM_CHECKPOINT`: Path to model checkpoint
- `FINLLM_INDEX`: Path to retrieval index

## Security Considerations

1. **Authentication**: Add authentication middleware for production
2. **HTTPS**: Use SSL/TLS certificates (Let's Encrypt recommended)
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **CORS**: Configure CORS headers appropriately
5. **Input Validation**: All user inputs are validated server-side

## Monitoring

Monitor these metrics in production:
- Request latency
- Model inference time
- Memory usage
- Error rates
- Active connections

## Scaling

For high-traffic deployments:
1. Use multiple worker processes
2. Deploy behind a load balancer
3. Cache retrieval results
4. Consider GPU acceleration for model inference

## Troubleshooting

### Port Already in Use
```bash
# Find process using the port
lsof -i :{port}
# Kill the process
kill -9 <PID>
```

### Model Not Loading
- Verify checkpoint path exists
- Check file permissions
- Ensure sufficient memory

### Retrieval Index Missing
```bash
python scripts/build_retrieval_index.py \\
  --corpus financial_training_corpus_clean.txt \\
  --index data/retrieval/finance_fts.sqlite
```

## Support

For issues and questions:
- Check the main README.md
- Review MODEL_CARD.md for model details
- Inspect logs in the application directory

## License

See LICENSE file in the project root.
"""
    
    docs_path = output_dir / "DEPLOYMENT.md"
    docs_path.write_text(docs_content)
    print(f"✓ Deployment documentation created at {docs_path}")


def create_run_script(output_dir: Path, host: str, port: int) -> None:
    """Create a simple run script."""
    # Unix/Linux script
    unix_script = f"""#!/bin/bash
set -e

echo "Starting FinLLM Studio..."
python webapp/server.py --host {host} --port {port}
"""
    
    unix_path = output_dir / "run.sh"
    unix_path.write_text(unix_script)
    unix_path.chmod(0o755)
    print(f"✓ Unix run script created at {unix_path}")
    
    # Windows script
    windows_script = f"""@echo off
echo Starting FinLLM Studio...
python webapp/server.py --host {host} --port {port}
"""
    
    windows_path = output_dir / "run.bat"
    windows_path.write_text(windows_script)
    print(f"✓ Windows run script created at {windows_path}")


def build_docker_image(output_dir: Path) -> None:
    """Build Docker image."""
    print("Building Docker image...")
    try:
        subprocess.run(
            ["docker", "build", "-t", "finllm-studio", "."],
            cwd=output_dir,
            check=True
        )
        print("✓ Docker image built successfully")
        print("\nTo run the container:")
        print("  docker run -p 8000:8000 finllm-studio")
    except subprocess.CalledProcessError as e:
        print(f"✗ Docker build failed: {e}")
    except FileNotFoundError:
        print("✗ Docker not found. Install Docker to build images.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    
    print("=" * 60)
    print("FinLLM Studio Deployment Builder")
    print("=" * 60)
    
    # Create deployment bundle
    create_deployment_bundle(output_dir)
    
    # Create Dockerfile
    create_dockerfile(output_dir, args.host, args.port)
    
    # Create documentation
    create_deployment_docs(output_dir, args.host, args.port)
    
    # Create run scripts
    create_run_script(output_dir, args.host, args.port)
    
    # Build Docker image if requested
    if args.docker:
        build_docker_image(output_dir)
    
    print("\n" + "=" * 60)
    print("Deployment bundle ready!")
    print("=" * 60)
    print(f"\nLocation: {output_dir}")
    print(f"\nTo deploy locally:")
    print(f"  cd {output_dir}")
    print(f"  ./run.sh  (Unix/Linux/Mac)")
    print(f"  run.bat   (Windows)")
    print(f"\nTo deploy with Docker:")
    print(f"  cd {output_dir}")
    print(f"  docker build -t finllm-studio .")
    print(f"  docker run -p {args.port}:{args.port} finllm-studio")
    print(f"\nSee {output_dir}/DEPLOYMENT.md for full instructions")


if __name__ == "__main__":
    main()
