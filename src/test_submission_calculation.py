import unittest
from src.solution import Solution

class TestSubmissionCalculations(unittest.TestCase):

    # Test 1: for sample_submission.csv
    def test_sample_submission(self):
        # Get solution
        sol = Solution(solution_input='data/sample_submission.csv')

        # Expected result
        expected_result = 144525525772.40201

        # Calculate the actual result
        actual_result = sol.calculate_wrw()

        # Check if the actual result matches the expected result
        self.assertAlmostEqual(actual_result, expected_result, places=2)

    # Test 2: for my_solution.csv in the solutions folder
    def test_my_solution(self):
        # Get solution
        sol = Solution(solution_input='solutions/sol1.csv')

        # Expected result
        expected_result = 12545621645.65260

        # Calculate the actual result
        actual_result = sol.calculate_wrw()

        # Check if the actual result matches the expected result
        self.assertAlmostEqual(actual_result, expected_result, places=2)

if __name__ == '__main__':
    unittest.main()