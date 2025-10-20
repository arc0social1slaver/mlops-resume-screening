from .util import prediction
import unittest


class Prediction(unittest.TestCase):

    def test_001(self):
        self.assertEqual(prediction("Data_Science.txt"), "Data Science")

    def test_002(self):
        self.assertEqual(prediction("Jane_Smith.txt"), "Health and fitness")

    def test_003(self):
        self.assertEqual(prediction("John_Doe.txt"), "Network Security Engineer")

    def test_004(self):
        self.assertEqual(prediction("Sarah_Williams.txt"), "Advocate")


if __name__ == "__main__":
    unittest.main()
