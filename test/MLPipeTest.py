import unittest 
from MLPipe.Pipe import pipeline_model
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

class TestPipeline(unittest.TestCase): 

    def test_pipeline_return(self): 

        cv = KFold(n_splits = 5, random_state=42, shuffle=True)

        pipe = pipeline_model(cv) 

        self.assertEqual(pipe, Pipeline)



if __name__ == "__main__": 
    unittest.main() 