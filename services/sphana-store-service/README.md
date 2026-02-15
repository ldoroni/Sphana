# Get Started

## Install Locally
Run the following commands:
1. Install Python Dependencies:<br>
   `uv sync --project .\services\sphana-store-service`
2. Activate the Virtual Environment:<br>
   `.\services\sphana-store-service\.venv\Scripts\activate`

## Refresh Dependencies
```
uv pip install -e .\services\sphana-store-service\ --refresh --reinstall 
```

## Run Locally
```
uv run python .\services\sphana-store-service\src\python\sphana_store\
```

## Build as Binary (in C lang)
Run the following command:
```
python -m nuitka `
   --mingw64 `
   --standalone `
   --remove-output `
   --prefer-source-code `
   --lto=no `
   --module-parameter=torch-disable-jit=yes `
   --python-flag=no_docstrings `
   --noinclude-pytest-mode=nofollow `
   --output-dir=.\services\sphana-store-service\target `
   --include-data-dir=.\services\sphana-store-service\src\resources=resources `
   .\services\sphana-store-service\src\python\sphana_store
```

### Notes
- Nuitka converts the python code into C, and and compile it as .exe file
  The .exe has ~20% better performance than the original Python code, but it still significantly slower than .NET Core 10
  It has security benefits, as it makes it harder to Reverse Engineer it, compared to regular Python code
- The main.exe file along with the required dlls, will be available under .\target\main.dist directory 
- "--mingw64" downloads the required C compiler automatically
- "--onefile" compiles the whole code into a single .exe file, which prevents exposure of Python libraries
- "--remove-output" cleanup intermadiate compilation files
- "--no-debug-symbols" rename classes and method to unclear names, but in Linux only 
  e.g. validate_admon_password(...) can ne renamed to something like sub_401240(...)
  This option will make stack traces in case of error to be meaningless, as the classes and the method names has changed.
- "--lto=yes" mixing the code of all modules together, in a way that it become harder to separate them to their original modules
  This option makes the .exe binary smaller and faster, but it has 0 impact on the logs.
- "--prefer-source-code" makes Nuitka to prefer original source code over .pyc files.
- "--unstripped" ensure to NOT use it!
<!-- - "--nofollow-import-to" makes nuitka to stop crashing inside the torch library.
   You should tell it not to compile the heavy ML libraries into the binary. 
   These libraries are already highly optimized shared objects (.so files).
   However, the disadvantage of this solution is that it will force installing python in the OS! -->

### Docker Build
Run the following command:
1. No CUDA:
   `docker build --no-cache -t sphana.store:1.0.0 -f .\services\sphana-store-service\src\docker\DockerFile .`
2. CUDA 12.8 Support:
   `docker build --no-cache -t sphana.store:1.0.0-cuda12.8 -f .\services\sphana-store-service\src\docker\Dockerfile-CUDA12.8 .`
