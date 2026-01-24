# Get Started

## Install Locally
Run the following commands:
1. Install Python Dependencies:<br>
   `uv sync --project .\libraries\request-handler-lib`
2. Activate the Virtual Environment:<br>
   `.\libraries\request-handler-lib\.venv\Scripts\activate`

## Build Library
Run the following command:
1. Build library:<br>
   `uv build --project .\libraries\request-handler-lib`
2. Publish to repository:<br>
   `uv publish --publish-url http://localhost:61000/ .\libraries\request-handler-lib\dist\*`