# Get Started

## Install Locally
Run the following commands:
1. Install Python Dependencies:<br>
   `uv sync --project .\libraries\client-handler-lib`
2. Activate the Virtual Environment:<br>
   `.\libraries\client-handler-lib\.venv\Scripts\activate`

## Build Library
Run the following command:
1. Build library:<br>
   `uv build --project .\libraries\client-handler-lib`
2. Publish to repository:<br>
   `uv publish --publish-url http://localhost:61000/  .\libraries\client-handler-lib\dist\*`