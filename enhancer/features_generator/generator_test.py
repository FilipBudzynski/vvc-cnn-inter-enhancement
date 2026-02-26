import unittest
from features_generator.generator import FeatureMapGenerator
from features_parser.tokens import MotionVector, ScalarToken, VectorToken


class TestFeatureMapGenerator(unittest.TestCase):
    def setUp(self):
        self.width = 16
        self.height = 16
        self.generator = FeatureMapGenerator(self.width, self.height)

    def test_generate_scalar_map(self):
        tokens = [ScalarToken(poc=0, x=8, y=8, w=8, h=8, param="QP", value=25.0)]

        maps = self.generator.generate_maps_for_frame(tokens)

        self.assertIn("QP", maps)
        self.assertEqual(maps["QP"].shape, (16, 16))
        self.assertEqual(maps["QP"][12, 12], 25.0)
        self.assertEqual(maps["QP"][8, 8], 25.0)
        self.assertEqual(maps["QP"][15, 15], 25.0)
        self.assertEqual(maps["QP"][0, 0], 0.0)

    def test_generate_vector_map_splitting(self):
        tokens = [
            VectorToken(
                poc=0,
                x=0,
                y=0,
                w=4,
                h=4,
                param="MVL0",
                value=MotionVector(x=-2.5, y=1.0),
            )
        ]

        maps = self.generator.generate_maps_for_frame(tokens)

        self.assertIn("MVL0_X", maps)
        self.assertIn("MVL0_Y", maps)

        self.assertEqual(maps["MVL0_X"][2, 2], -2.5)
        self.assertEqual(maps["MVL0_Y"][2, 2], 1.0)
