class GraphError(Exception):
    """
    Exception class for the Graph class

    """
    def __init__(self, msg=""):
        super().__init__(msg)