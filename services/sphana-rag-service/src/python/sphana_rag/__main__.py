from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sphana_rag.controllers.indices.v1 import index_management_controller_router
import uvicorn

class Application:
    def __init__(self):
        # Create FastAPI app
        self.app = FastAPI(
            title="Sphana RAG API",
            description="API for Sphana RAG",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.include_router(index_management_controller_router)

    def run(self, host='0.0.0.0', port=5001, debug=False):
        # logger.info(f"Starting FastAPI server on {host}:{port}")
        # logger.info("Available endpoints:")
        # logger.info("- POST /generate - Send a prompt and get an answer")
        # logger.info("- POST /generate:stream - Send a prompt and get streaming answer (Server-Sent Events)")
        # logger.info("- POST /start_session - Start a new session")
        # logger.info("- GET /docs - Interactive API documentation")
        # logger.info("- GET /redoc - Alternative API documentation")
        uvicorn.run(self.app, host=host, port=port, reload=debug)

if __name__ == "__main__":
    app = Application()
    app.run()