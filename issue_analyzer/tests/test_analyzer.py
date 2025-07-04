import unittest
from src.issue_analyzer.analyzer import IssueAnalyzer
from src.issue_analyzer.schema import IssueOutput

class TestIssueAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = IssueAnalyzer()

    def test_valid_issue(self):
        text = "A fire broke out at the Petrobras refinery in Cubatão on July 2nd."
        result = self.analyzer.analyze(text)
        self.assertIsInstance(result, IssueOutput)
        self.assertTrue(result.is_issue)
        self.assertIsNotNone(result.issue_analysis)
        self.assertEqual(result.meta.language, "en")

    def test_non_issue(self):
        text = "A new cultural exhibition opened at the Museum of Art in São Paulo."
        result = self.analyzer.analyze(text)
        self.assertIsInstance(result, IssueOutput)
        self.assertFalse(result.is_issue)
        self.assertIsNone(result.issue_analysis)

if __name__ == "__main__":
    unittest.main()
