import unittest
from features_parser.parser import VTMParser


class TestVTMParser(unittest.TestCase):
    def setUp(self):
        self.parser = VTMParser()
        self.log_content = [
            "BlockStat: POC 31 @(   0,   0) [ 8x 4] QP=18\n",
            "BlockStat: POC 31 @(   8,   0) [ 8x 8] MVL0={-4, 12}\n",
            "Irrelevant line\n",
            "BlockStat: POC 32 @(  16,  16) [ 16x 16] Depth=3\n",
        ]

    def test_parser_integration(self):
        self.parser.parse(self.log_content)

        self.assertEqual(len(self.parser.tokens), 3)
        qp = self.parser.tokens[0]
        self.assertEqual(qp.param, "QP")
        self.assertEqual(qp.value, 18.0)

        mv = self.parser.tokens[1]
        self.assertEqual(mv.param, "MVL0")
        self.assertEqual(mv.value, (-4.0, 12.0))

        depth = self.parser.tokens[2]
        self.assertEqual(depth.poc, 32)

    def test_invalid_line_handling(self):
        garbage = ["Random string\n", "Another one\n"]
        self.parser.parse(garbage)
        self.assertEqual(len(self.parser.tokens), 0)

    def test_scalar_parsing(self):
        self.parser.parse(self.log_content)

        qp_token = self.parser.tokens[0]
        self.assertEqual(qp_token.param, "QP")
        self.assertEqual(qp_token.value, 18.0)
        self.assertEqual(qp_token.x, 0)
        self.assertEqual(qp_token.y, 0)
        self.assertEqual(qp_token.w, 8)
        self.assertEqual(qp_token.h, 4)

    def test_vector_parsing(self):
        self.parser.parse(self.log_content)

        token = self.parser.tokens[1]
        self.assertIsNotNone(token)
        self.assertEqual(token.param, "MVL0")
        self.assertEqual(token.value, (-4.0, 12.0))
        self.assertEqual(token.poc, 31)
        self.assertEqual(token.x, 8)
        self.assertEqual(token.w, 8)
