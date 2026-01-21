### Pre-Requisite
1. Install Python 3.12
2. Install UV (for pyproject.toml installations)
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

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
   `uv sync --project .\libraries\request-handler-lib`
2. Activate the Virtual Environment:<br>
   `.\libraries\request-handler-lib\.venv\Scripts\activate`

### Build Library
Run the following command:
1. Build library:<br>
   `uv build --project .\libraries\request-handler-lib`
2. Publish to repository:<br>
   `uv publish --publish-url http://localhost:61000/  .\libraries\request-handler-lib\dist\*`