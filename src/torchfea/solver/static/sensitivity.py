from ..basesensitivity import BaseSensitivityAnalyzer

class StaticSensitivityAnalyzer(BaseSensitivityAnalyzer):
    
    def get_sensitivity(self):
        """
        get the sensitivity of the given response (a function of the displacement and design variables) with respect to the design variables.
        """
        
        
