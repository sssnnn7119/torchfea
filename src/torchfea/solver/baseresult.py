

class BaseResult():
    """
    A class to represent the results of a finite element analysis (FEA) simulation.
    """

    def save(self, path: str):
        """
        Save the FEA result to a file.

        Args:
            path (str): The path to the file where the result will be saved.
        """
        pass

    @classmethod
    def load(cls, path: str) -> "BaseResult":
        """
        Load the FEA result from a file.

        Args:
            path (str): The path to the file from which the result will be loaded.
        """
        pass

    

