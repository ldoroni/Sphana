### Pre-Requisite
1. Install Python 3.12
2. Install UV (for pyproject.toml installations)
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   <!-- $env:Path = "C:\Users\ldoro\.local\bin;$env:Path" -->

# Create Local Repository
Run the following commands:
1. Install the server tool:
   `uv tool install pypiserver`
2. Create the local repository dir:
   `mkdir C:\Users\ldoro\.pyrepo`
3. Start the local repository:
   `pypi-server run -a . -P . -p 61000 --overwrite C:\Users\ldoro\.pyrepo`
4. Update or create the uv.toml file:
   `C:\Users\ldoro\AppData\Roaming\uv\uv.toml`
5. Write the following with in uv.toml file:
   ```
   [[index]]
   name = "local-repo"
   url = "http://localhost:61000"
   allow-insecure-host = ["0.0.0.0"]
   ```

### Install Locally
Run the following commands:
1. Install Python Dependencies:<br>
   `uv sync --project .\services\sphana-rag-service`
2. Activate the Virtual Environment:<br>
   `.\services\sphana-rag-service\.venv\Scripts\activate`

### Flow to update library:
`uv build --project .\libraries\managed-exceptions-lib`
`uv publish --publish-url http://localhost:61000/ .\libraries\managed-exceptions-lib\dist\*`
`uv pip install -e .\services\sphana-rag-service\ --refresh --reinstall`
`uv sync --project .\services\sphana-rag-service`
`python .\services\sphana-rag-service\src\python\`

### Build as Binary (in C lang)
Run the following command:
```
python -m nuitka `
    --mingw64 `
    --standalone `
    --include-data-dir=.\python\resources=resources `
    --output-dir=.\target `
    .\src\python\main.py
```

# Notes
- The main.exe file along with the required dlls, will be available under .\target\main.dist directory 
- "--mingw64" downloads the required C compiler automatically

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

