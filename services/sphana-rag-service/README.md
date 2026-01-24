# Get Started

## Install Locally
Run the following commands:
1. Install Python Dependencies:<br>
   `uv sync --project .\services\sphana-rag-service`
2. Activate the Virtual Environment:<br>
   `.\services\sphana-rag-service\.venv\Scripts\activate`

## Refresh Dependencies
```
uv pip install -e .\services\sphana-rag-service\ --refresh --reinstall 
```

## Run Locally
```
uv run python .\services\sphana-rag-service\src\python\sphana_rag\
```

## Build as Binary (in C lang)
Run the following command:
```
python -m nuitka `
    --mingw64 `
    --standalone `
    --onefile `
    --remove-output `
    --prefer-source-code `
    --lto=yes `
    --include-data-dir=.\services\sphana-rag-service\src\resources=resources `
    --output-dir=.\services\sphana-rag-service\target `
    .\services\sphana-rag-service\src\python\sphana_rag
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

### Docker Build
Run the following command:
1. No CUDA:
   `docker build --no-cache -t sphana.rag:1.0.0 -f .\services\sphana-rag-service\src\docker\DockerFile .`
2. CUDA 12.8 Support:
   `docker build --no-cache -t sphana.rag:1.0.0-cuda12.8 -f .\services\sphana-rag-service\src\docker\DockerFile-CUDA12.8 .`

