"""
Test script for predict_handler.py
"""
import os
import unittest
import warnings
import shutil

import predict_handler

class TestMetricsDataHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore", ResourceWarning)
        try:
            os.mkdir('./tmp')
        except FileExistsError:
            pass
    
    def test_process_file(self):
        path_to_zip_file = "./test.zip"
        predict_handler.extract_datasets(path_to_zip_file)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(os.environ["FOLDER_PATH"])

if __name__ == '__main__':
    unittest.main()