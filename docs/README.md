# Get Started

## Pre-Requisite

<!-- Install c++:
https://visualstudio.microsoft.com/visual-cpp-build-tools/
1. Select Desktop development with C++.
2. Under Optional Components, check:
2.1. Windows 10/11 SDK (or the latest version listed).
2.2. If working with legacy code, you can also select older SDKs (e.g., Windows 8.1 SDK), but this is rarely needed. -->

1. Install Python 12:
   As PyTorch requires Python v3.9.x-3.12.x (https://pytorch.org/get-started/locally/)
   https://www.python.org/downloads/release/python-31210/

2. Install UV (for pyproject.toml installations):
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

3. Install CUDA 12.8:
   https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#id6
   https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

4. Install NVidia Studio Driver 572.83:
   https://www.nvidia.com/en-us/drivers/details/242208/

5. Install cuDNN 9.8.0
   https://docs.nvidia.com/deeplearning/cudnn/backend/v9.8.0/reference/support-matrix.html
   https://developer.nvidia.com/cudnn-9-8-0-download-archive

## Create Local PyPi Repository
Run the following commands:
1. Install the server tool:
   `uv tool install pypiserver`
2. Create the local repository dir:
   `mkdir C:\Users\ldoro\.pyrepo`
3. Update or create the uv.toml file:
   `C:\Users\ldoro\AppData\Roaming\uv\uv.toml`
4. Write the following with in uv.toml file:
   ```
   [[index]]
   name = "local-repo"
   url = "http://localhost:61000"
   allow-insecure-host = ["0.0.0.0"]
   ```
5. Start the local repository:
   `pypi-server run -a . -P . -p 61000 --overwrite C:\Users\ldoro\.pyrepo`

## Example for Update Library Flow:
1. Build Library:
   `uv build --project .\libraries\managed-exceptions-lib`
2. `uv publish --publish-url http://localhost:61000/ .\libraries\managed-exceptions-lib\dist\*`
3. `uv pip install -e .\services\sphana-rag-service\ --refresh --reinstall`
4. `uv sync --project .\services\sphana-rag-service`
5. `python .\services\sphana-rag-service\src\python\`


# Notes
- We are using 'nuitka' to compile the python application as C.
  The 'nuitka' library currently requires Python v12 only!
  It cannot work with newer versions.