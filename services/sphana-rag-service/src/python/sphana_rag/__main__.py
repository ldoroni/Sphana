import logging
import uvicorn
from injector import Injector
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_injector import attach_injector
from sphana_rag.controllers.documents.v1 import router as document_management_controller_router
from sphana_rag.controllers.indices.v1 import router as index_management_controller_router
from sphana_rag.controllers.queries.v1 import router as query_executor_controller_router
from request_handler import RequestThreadPool

def main(host='0.0.0.0', port=5001, max_threads=100, debug=False):
    #####################
    # Configure Logging #
    #####################

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: [%(name)s] %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%SZ',
        handlers=[logging.StreamHandler()]
    )

    ##############################
    # Initialise API Thread Pool #
    ##############################

    RequestThreadPool.init(max_workers=max_threads)

    ######################
    # Initialize FastAPI #
    ######################

    # Initialize FastAPI
    fast_api: FastAPI = FastAPI(
        title="Sphana RAG API",
        description="API for Sphana RAG",
        version="1.0.0"
    )
    
    # Add CORS middleware
    fast_api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    fast_api.include_router(document_management_controller_router)
    fast_api.include_router(index_management_controller_router)
    fast_api.include_router(query_executor_controller_router)

    ##################################
    # Initialize Dependency Injector #
    ##################################

    # Initialize the dependency injector
    injector: Injector = Injector()
    attach_injector(fast_api, injector)

    ################
    # Start Server #
    ################
    
    # Start the server
    uvicorn.run(fast_api, host=host, port=port, reload=debug, access_log=debug)

if __name__ == "__main__":
    main()
