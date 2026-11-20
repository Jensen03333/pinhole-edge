import json
import unittest

import numpy as np

from baseline import expand
from detect import decode, letterbox
from metrics import ap50, at_threshold, match_image, pick_threshold
from platform_lab import controls, measure
from simulate_cell import Cell
from vision_io import READY, yolo_box_pixels


class Boxes(unittest.TestCase):
    def test_published_box(self):
        b = yolo_box_pixels(.228125, .807291667, .028125, .035416667, 640, 480)
        np.testing.assert_allclose(b, [137,379,155,396], atol=.00001)

    def test_clip_to_source(self):
        self.assertEqual(yolo_box_pixels(0,0,.5,.5,40,20), [0,0,10,5])
        self.assertEqual(expand([[0,0,39,18,.8]], 4, (20,40)), [[0,0,40,20,.8]])

    def test_letterbox_keeps_rgb_and_geometry(self):
        img = np.zeros((480,640,3), np.uint8)
        img[:] = [0,64,255]
        x, trans = letterbox(img,512)
        self.assertEqual(trans,(.8,.8,0,64))
        self.assertEqual(x.shape,(1,3,512,512))
        np.testing.assert_allclose(x[0,:,240,200], [1,64/255,0], atol=1e-7)
        np.testing.assert_allclose(x[0,:,0,0], [114/255]*3, atol=1e-7)

    def test_nms_clipping_and_threshold(self):
        # Two copies of one real target, one box fully in the letterbox margin and one sub-threshold.
        raw = np.array([[80,144,16,16,.8],[80,144,16,16,.6],[20,5,4,2,.99],[240,220,20,8,.2]],
                       dtype=np.float32).T[None]
        boxes = decode(raw,(.8,.8,0,64),(480,640),.4)
        self.assertEqual(len(boxes),1)
        np.testing.assert_allclose(boxes[0][:4], [90,90,110,110])

    def test_one_to_one_and_operating_point(self):
        truth = [[0,0,8,8],[20,20,30,30]]
        boxes = [[0,0,8,8,.9],[0,0,8,8,.7],[20,20,30,30,.6]]
        hits = match_image(boxes, truth)
        self.assertEqual([i for _,i in hits], [0,-1,1])
        rows = [({"boxes":truth}, hits), ({"boxes":[]}, [(.65,-1)])]
        report = at_threshold(rows,.6)
        self.assertEqual((report["tp"],report["fp"],report["fn"],report["no_pinhole_flagged"]), (2,2,0,1))
        self.assertGreater(ap50(rows), .5)
        self.assertEqual(pick_threshold(rows,[.9,.7])[1], .9)

    def test_blob_recipe_edges(self):
        expected = {"control_threshold":(2,32), "control_diagonal":(1,8),
                    "control_border_area":(1,4), "control_empty":(0,0)}
        for key, source in controls(38).items():
            _, boxes = measure(source,38)
            self.assertEqual((len(boxes),sum(b["area_px"] for b in boxes)),expected[key])


class Station(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def inspect(ident):
            if ident == "broken":
                raise ValueError("Invalid image")
            if ident == "engine_fault":
                raise RuntimeError("provider failed")
            self.calls.append(ident)
            return {"image":ident, "boxes":[[0,0,3,3,.9]] if ident=="pin" else []}
        self.cell = Cell(inspect, 100, 60)

    def event(self, t, typ, part, **kw):
        self.cell.step(dict(t=t,type=typ,part=part,**kw))

    def test_interleaved_and_duplicates(self):
        self.event(0,"trigger","A")
        self.event(1,"trigger","B")
        self.event(2,"trigger","A")
        self.event(3,"frame","A",tag="B",image="clear")
        self.event(4,"frame","B",tag="B",image="clear")
        self.event(5,"frame","A",tag="A",image="pin")
        self.event(6,"frame","A",tag="A",image="pin")
        self.event(7,"ack","A")
        self.event(8,"ack","B")
        self.assertEqual(self.calls,["clear","pin"])
        self.assertEqual(self.cell.parts["B"]["decision"],"NO_PINHOLE")
        self.assertEqual(self.cell.parts["A"]["decision"],"NG_PINHOLE")
        self.assertTrue(all(p["state"]=="ACKED" for p in self.cell.parts.values()))

    def test_timeout_exactly_at_deadline_is_closed(self):
        self.event(0,"trigger","A")
        self.event(100,"frame","A",tag="A",image="clear")
        self.event(160,"ack","A")
        self.assertFalse(self.calls)
        self.assertEqual(self.cell.parts["A"]["decision"],"REVIEW")
        self.assertEqual(self.cell.parts["A"]["state"],"HOLD_ACK_TIMEOUT")

    def test_bad_image_one_decision_only(self):
        self.event(0,"trigger","A")
        self.event(1,"frame","A",tag="A",image="broken")
        self.event(2,"frame","A",tag="A",image="clear")
        self.assertEqual(self.cell.parts["A"]["reason"],"image_error")
        self.assertEqual(sum(e["event"]=="result" for e in self.cell.events),1)

    def test_unknown_and_backward_events(self):
        self.event(5,"ack","U")
        self.event(8,"frame","U",tag="U",image="clear")
        self.assertFalse(self.cell.parts)
        with self.assertRaises(ValueError):
            self.cell.advance(2)

    def test_engine_failure_goes_to_review(self):
        self.event(0,"trigger","A")
        self.event(2,"frame","A",tag="A",image="engine_fault")
        self.assertEqual(self.cell.parts["A"]["decision"],"REVIEW")
        self.assertEqual(self.cell.parts["A"]["reason"],"inference_error")


@unittest.skipUnless((READY/"audit.json").is_file(),"fetch and prepare CoatingVision first")
class Dataset(unittest.TestCase):
    def test_sequences_pixels_and_annotations(self):
        rows = json.loads((READY/"manifest.json").read_text(encoding="utf-8"))
        seq = {s:{r["sequence"] for r in rows if r["split"]==s} for s in ("train","val","test")}
        self.assertFalse(seq["train"]&seq["val"] or seq["train"]&seq["test"] or seq["val"]&seq["test"])
        self.assertEqual(len(rows),2149)
        self.assertEqual(len({r["sha256_pixels"] for r in rows}),len(rows))
        self.assertEqual(sum(r["used_for_fit"] for r in rows),905)
        self.assertTrue(all((r["width"],r["height"])==(640,480) for r in rows))
        self.assertEqual(sum(len(r["boxes"]) for r in rows if r["split"]=="test"),60)
