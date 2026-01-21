# Get Started

## Install Locally
Run the following commands:
1. Install Python Dependencies:<br>
   `uv sync --project .\services\sphana-rag-service`
2. Activate the Virtual Environment:<br>
   `.\services\sphana-rag-service\.venv\Scripts\activate`

## Build as Binary (in C lang)
Run the following command:
```
python -m nuitka `
    --mingw64 `
    --standalone `
    --include-data-dir=.\services\sphana-rag-service\src\resources=resources `
    --output-dir=.\services\sphana-rag-service\target `
    .\services\sphana-rag-service\src\python\sphana_rag
```

<!-- # Notes
- The main.exe file along with the required dlls, will be available under .\target\main.dist directory 
- "--mingw64" downloads the required C compiler automatically -->

<!-- python -m nuitka `
    --standalone `
    --onefile `
    --plugin-enable=torch `
    --plugin-enable=numpy `
    --include-data-dir=./models=models `
    --output-dir=.\target `
    .\src\python\main.py -->

### Docker Build
Run the following command:
```
docker build --no-cache -t sphana.rag:1.0.0 -f .\src\docker\DockerFile .
```

