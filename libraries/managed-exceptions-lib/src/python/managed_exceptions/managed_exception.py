from managed_exceptions.error_details import ErrorDetails

class ManagedException(Exception):
    def __init__(self, error: ErrorDetails):
        self.status_code = error.status_code
        self.diagnostic_code = error.diagnostic_code
        self.diagnostic_details = error.diagnostic_details
        super().__init__(error.message)