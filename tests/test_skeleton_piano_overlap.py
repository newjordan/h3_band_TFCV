"""The fresh-hand overlap must not feed the previous H3 hands back to the VAE."""
import pathlib
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from h3_tools import skeleton_piano as piano


class OverlapPlateTest(unittest.TestCase):
    def compose(self, background_only):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            frames_dir = root / 'out' / 'sk_probe_c0'
            frames_dir.mkdir(parents=True)
            old = np.full((8, 12, 3), 200, dtype=np.uint8)
            for k in range(2):
                cv2.imwrite(str(frames_dir / f'frame_{k:05d}.png'), old)
            plates = [np.full_like(old, 20 + k) for k in range(4)]
            cap = mock.Mock()
            cap.read.side_effect = [(True, im.copy()) for im in plates] + [(False, None)]
            proc = mock.Mock()
            proc.wait.return_value = 0
            config = {'overlap': 2, 'bg_overlap': background_only}
            with (mock.patch.dict(piano.RUNS, {'probe': config}),
                  mock.patch.object(piano, 'COMFY_OUT', root / 'out'),
                  mock.patch.object(piano, 'COMFY_IN', root / 'in'),
                  mock.patch.object(piano, 'hands_box', return_value=[2, 1, 9, 6]),
                  mock.patch('cv2.VideoCapture', return_value=cap),
                  mock.patch.object(piano.subprocess, 'Popen', return_value=proc)):
                piano.overlap_plate('probe', 1)
            result = [np.frombuffer(c.args[0], dtype=np.uint8).reshape(old.shape)
                      for c in proc.stdin.write.call_args_list]
            return result, plates, old

    def test_background_only_keeps_hand_free_plate_inside_box(self):
        result, plates, old = self.compose(True)
        for k in range(2):
            np.testing.assert_array_equal(result[k][1:6, 2:9], plates[k][1:6, 2:9],
                                          err_msg='fresh box still contains previous H3 hands')
            np.testing.assert_array_equal(result[k][0], old[0])
            np.testing.assert_array_equal(result[k][:, 9:], old[:, 9:])
        np.testing.assert_array_equal(result[2:], plates[2:])

    def test_full_overlap_still_holds_previous_h3_everywhere(self):
        result, plates, old = self.compose(False)
        np.testing.assert_array_equal(result[:2], [old, old])
        np.testing.assert_array_equal(result[2:], plates[2:])


if __name__ == '__main__':
    unittest.main()
