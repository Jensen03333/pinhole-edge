"""Software-in-the-loop replay: file camera, real CPU inference and a virtual PLC handshake."""
import argparse
import json
from html import escape
from pathlib import Path
from time import perf_counter

from detect import Detector
from fetch_assets import digest
from vision_io import READY, ROOT, RUNS, annotate, read_image, write_image, write_json


class Cell:
    def __init__(self, inspect, frame_ms, ack_ms):
        self.inspect = inspect
        self.frame_ms = frame_ms
        self.ack_ms = ack_ms
        self.parts = {}
        self.events = []
        self.now = 0

    def record(self, part, event, **details):
        self.events.append(dict(t=self.now, part=part, event=event, **details))

    def result(self, p, decision, reason, **details):
        p.update(state="WAIT_ACK", decision=decision, reason=reason,
                 result_ms=self.now, due=self.now+self.ack_ms, **details)
        self.record(p["part"], "result", decision=decision, reason=reason, **details)

    def advance(self, moment):
        if moment < self.now:
            raise ValueError("Replay events must be chronological.")
        pending = [p for p in self.parts.values() if p["state"] in ("WAIT_FRAME", "WAIT_ACK")]
        while pending:
            p = min(pending, key=lambda row: row["due"])
            if p["due"] > moment:
                break
            self.now = p["due"]
            if p["state"] == "WAIT_FRAME":
                self.result(p, "REVIEW", "frame_timeout")
            else:
                p["state"] = "HOLD_ACK_TIMEOUT"
                self.record(p["part"], "ack_timeout")
                pending.remove(p)
        self.now = moment

    def step(self, e):
        self.advance(e["t"])
        ident, kind = e["part"], e["type"]
        p = self.parts.get(ident)
        if kind == "trigger":
            if p is not None:
                self.record(ident, "duplicate_trigger")
                return
            self.parts[ident] = dict(part=ident, state="WAIT_FRAME", decision=None,
                                     trigger_ms=self.now, due=self.now+self.frame_ms)
            self.record(ident, "trigger")
        elif kind == "frame":
            if p is None or p["state"] != "WAIT_FRAME":
                self.record(ident, "stale_or_unknown_frame")
                return
            if e["tag"] != ident:
                self.record(ident, "frame_tag_mismatch", received=e["tag"])
                return
            try:
                found = self.inspect(e["image"])
            except (OSError, ValueError) as err:
                self.result(p, "REVIEW", "image_error", message=type(err).__name__)
                return
            except RuntimeError as err:
                self.result(p, "REVIEW", "inference_error", message=str(err))
                return
            self.result(p, "NG_PINHOLE" if found["boxes"] else "NO_PINHOLE", "inference", **found)
        elif kind == "ack":
            if p is None or p["state"] != "WAIT_ACK":
                self.record(ident, "unexpected_ack")
                return
            p["state"] = "ACKED"
            self.record(ident, "ack")
        else:
            raise ValueError(f"Unknown event type {kind}")


def replay_html(cell, out):
    body = []
    for e in cell.events:
        note = e.get("decision", e.get("received", ""))
        if e.get("reason"):
            note += " / " + e["reason"]
        if "image" in e:
            note += f' · {e["image"]} · {len(e["boxes"])} 个预测框'
        photo = f'<a href="frames/{escape(e["image"])}.jpg">查看推理叠框</a>' if "image" in e else ""
        body.append(f'<tr><td>{e["t"]}</td><td>{escape(e["part"])}</td><td>{escape(e["event"])}</td>'
                    f'<td>{escape(note)} {photo}</td></tr>')
    final = ''.join(f'<li>{escape(p["part"])} · {escape(p["decision"])} · {escape(p["state"])}</li>'
                    for p in cell.parts.values())
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>电极涂布工位回放</title>'
            '<style>body{font:16px system-ui;color:#173039;max-width:940px;margin:2em auto;padding:0 1em}'
            'td,th{text-align:left;border-bottom:1px solid #cedbde;padding:.4em .7em}table{border-collapse:collapse}'
            'input{width:min(380px,70vw)}a{color:#075b60}button{padding:.4em 1em}</style>'
            '<h1>电极涂布针孔工位回放</h1><p>点击从头回放，或拖动时间轴。时间为虚拟毫秒，绿框来自本机 ONNX 推理。'
            'NO_PINHOLE 表示本帧没有检出针孔。HOLD_ACK_TIMEOUT 表示结果发出后未及时收到模拟 PLC 应答。</p>'
            f'<input id="cursor" type="range" min="0" max="{len(body)}" value="{len(body)}" aria-label="事件序号"> '
            '<button id="restart">从头回放</button> <button id="next">下一步</button> <output id="counter"></output>'
            '<table><thead><tr><th>毫秒</th><th>图块触发 ID</th><th>事件</th><th>结果</th></tr></thead>'
            '<tbody>'+''.join(body)+'</tbody></table><details><summary>最终状态</summary><ul>'+final+'</ul></details>'
            '<p>采集方式为文件回放，每个触发对应一个图块。相机、PLC 与执行机构均为软件替身。图像经叠框，来自 '
            '<a href="https://doi.org/10.6084/m9.figshare.29260121.v1">CoatingVision v1</a>，CC BY 4.0。</p>'
            '<script>const rows=[...document.querySelectorAll("tbody tr")],s=document.querySelector("#cursor");'
            'function draw(){rows.forEach((r,i)=>r.hidden=i>=Number(s.value));'
            'document.querySelector("#next").disabled=Number(s.value)>=rows.length;'
            'document.querySelector("#counter").textContent=s.value+" / "+rows.length;}'
            's.oninput=draw;document.querySelector("#restart").onclick=()=>{s.value=0;draw()};'
            'document.querySelector("#next").onclick=()=>{s.value=Math.min(rows.length,Number(s.value)+1);draw()};'
            'draw()</script></html>')
    (out / "replay.html").write_text(html, encoding="utf-8")


def run(path):
    recipe = json.loads(path.read_text(encoding="utf-8"))
    output = RUNS / "cell"
    model = RUNS / "edge/pinhole.onnx"
    settings = json.loads((RUNS / "edge/operating_point.json").read_text(encoding="utf-8"))
    if digest(model, "sha256") != settings["model_sha256"]:
        raise ValueError("Calibrated model checksum does not match.")
    engine = Detector(model)

    def inspect(ident):
        if not ident.startswith("image_") or not ident.removeprefix("image_").isdigit():
            raise ValueError("Invalid fixture ID")
        im = read_image(READY / "images/val" / (ident + ".jpg"))
        start = perf_counter()
        try:
            boxes = engine.predict(im, settings["confidence"])
        except Exception as err:
            # Providers use distinct exception classes. A failed frame must always reach REVIEW.
            raise RuntimeError(type(err).__name__) from err
        elapsed = round((perf_counter()-start)*1000, 3)
        write_image(output / "frames" / (ident+".jpg"), annotate(im, boxes, (45,220,75)))
        return dict(image=ident, boxes=boxes, inference_ms=elapsed)

    cell = Cell(inspect, recipe["frame_timeout_ms"], recipe["ack_timeout_ms"])
    for e in recipe["events"]:
        cell.step(e)
    cell.advance(recipe["stop_ms"])
    report = dict(model_sha256=settings["model_sha256"], virtual_deadlines=True,
                  parts=list(cell.parts.values()), events=cell.events)
    write_json(output / "trace.json", report)
    replay_html(cell, output)
    for p in cell.parts.values():
        print(p["part"], p["decision"], p["state"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=ROOT / "simulation/scenario.json")
    run(parser.parse_args().recipe)
