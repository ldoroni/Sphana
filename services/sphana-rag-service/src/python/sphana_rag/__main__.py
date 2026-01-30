import uvicorn
from injector import Injector
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_injector import attach_injector
from sphana_rag.controllers.documents.v1 import router as document_management_controller_router
from sphana_rag.controllers.indices.v1 import router as index_management_controller_router
from sphana_rag.controllers.queries.v1 import router as query_executor_controller_router

def main(host='0.0.0.0', port=5001, debug=False):
    ##########################
    # Initialize FastAPI app #
    ##########################

    # Create FastAPI app
    app: FastAPI = FastAPI(
        title="Sphana RAG API",
        description="API for Sphana RAG",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(document_management_controller_router)
    app.include_router(index_management_controller_router)
    app.include_router(query_executor_controller_router)

    ##################################
    # Initialize Dependency Injector #
    ##################################

    # Initialize the dependency injector
    injector: Injector = Injector()
    attach_injector(app, injector)

    ################
    # Start Server #
    ################
    
    # Start the server
    uvicorn.run(app, host=host, port=port, reload=debug)

if __name__ == "__main__":
    main()