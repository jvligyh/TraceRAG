class DocumentInputError(ValueError):
    """只有应用自己构造的提示可原样交给用户，第三方异常仍隐藏细节。"""

    def __init__(self, message: str, code: str = "invalid_document", status_code: int = 400):
        super().__init__(message)
        self.public_message=message
        self.code=code
        self.status_code=status_code
