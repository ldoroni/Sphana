# Get Started

## Pre-Requisite
1. Install Python 3.12
2. Install UV (for pyproject.toml installations)
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Create Local Repository
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

## Example for Update Library Flow:
1. Build Library:
   `uv build --project .\libraries\managed-exceptions-lib`
2. `uv publish --publish-url http://localhost:61000/ .\libraries\managed-exceptions-lib\dist\*`
3. `uv pip install -e .\services\sphana-rag-service\ --refresh --reinstall`
4. `uv sync --project .\services\sphana-rag-service`
5. `python .\services\sphana-rag-service\src\python\`
