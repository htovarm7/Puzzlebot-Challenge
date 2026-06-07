#!/usr/bin/env python3
"""
intersection_detector.py  (v3 — Inverse Perspective Mapping + dash grouping)

Detects DASHED intersection markings and reports which arms exist
(front / back / left / right), separately from the line-following module.

Pipeline
--------
1. IPM warp        : perspective transform from 4 floor points -> bird's-eye.
                     Dashes become uniform; forward/left/right become
                     orthogonal; an intersection becomes a square, not a
                     trapezoid. (Toggle with ipm_enable; falls back to an
                     ROI crop on the raw frame when off.)
2. Threshold+morph : adaptive threshold, opening/closing to clean speckle.
3. Segments        : contours filtered by area + aspect -> dash blobs.
4. Grouping        : collinear blobs merged into candidate lines.
5. Dashed vs solid : fill_ratio (covered/span) + straightness residual.
                       dashed -> gaps, low fill, straight.
                       solid  -> fills its span (~1.0)  -> rejected.
                       curve  -> high residual          -> rejected.
6. Arms            : in the warped view, vertical lines in the centre =>
                     forward/back; horizontal lines reaching the side
                     zones => left/right.
7. Debounce        : in the node, confirm over several frames.

Topics (unchanged)
------------------
Sub : /camera/image_raw       (sensor_msgs/Image)
Pub : /intersection/detected  (std_msgs/Bool)
      /intersection/arms       (std_msgs/String)   "front,left,right" | "none"
      /vision/intersection     (sensor_msgs/Image)  warped annotated view
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge


ARM_ORDER = ("front", "back", "left", "right")

_DEFAULT_PARAMS = {
    # ── Inverse Perspective Mapping (bird's-eye) ───────────────────────
    "ipm_enable":         1,
    "src_top_x":          0.30,   # top edge spans x in [src_top_x, 1-src_top_x]
    "src_top_y":          0.62,   # y of the FAR (top) edge of the road quad
    "src_bot_x":          0.02,   # bottom edge spans x in [src_bot_x, 1-src_bot_x]
    "src_bot_y":          0.98,   # y of the NEAR (bottom) edge
    "warp_w":             360,    # bird's-eye output width
    "warp_h":             480,    # bird's-eye output height

    # ── Fallback ROI (used only when ipm_enable = 0) ───────────────────
    "roi_top":            0.45,

    # ── Threshold + morphology ─────────────────────────────────────────
    "blur":               5,
    "adapt_block":        41,
    "adapt_c":            10,
    "morph":              3,

    # ── Per-segment filtering ──────────────────────────────────────────
    "seg_min_area":       40.0,
    "seg_max_area":       6000.0,
    "seg_min_aspect":     1.5,

    # ── Grouping collinear segments ────────────────────────────────────
    "group_angle_tol":    18.0,
    "group_perp_tol":     22.0,
    "min_dashes_in_line": 4,

    # ── DASHED vs SOLID / curve discrimination ─────────────────────────
    "min_fill_ratio":     0.12,
    "max_fill_ratio":     0.88,
    "max_resid_px":       18.0,

    # ── Zones (fractions of the WORK image: warped, or ROI if ipm off) ─
    "left_edge":          0.33,
    "right_edge":         0.66,
    "front_edge":         0.45,
    "back_edge":          0.85,
    "horiz_vert_split":   45.0,

    # ── Decision / stabilisation ───────────────────────────────────────
    "min_lines_per_arm":  1,
    "debounce":           3,
}


def _load_params_yaml(path: str) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
        out = dict(_DEFAULT_PARAMS)
        out.update({k: v for k, v in raw.items() if k in _DEFAULT_PARAMS})
        return out
    except Exception:
        return None


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def src_quad(frame_shape, params) -> np.ndarray:
    """The 4 source floor points (TL, TR, BR, BL) in raw-image pixels."""
    H, W = frame_shape[:2]
    stx = float(params["src_top_x"]) * W
    sty = float(params["src_top_y"]) * H
    sbx = float(params["src_bot_x"]) * W
    sby = float(params["src_bot_y"]) * H
    return np.float32([[stx, sty], [W - stx, sty],
                       [W - sbx, sby], [sbx, sby]])


def draw_src_quad(frame: np.ndarray, params: dict) -> np.ndarray:
    """For the calibrator: show the IPM source region on the raw frame."""
    vis = frame.copy()
    quad = src_quad(frame.shape, params).astype(int)
    cv2.polylines(vis, [quad], True, (0, 255, 255), 2)
    for (x, y), lbl in zip(quad, ("TL", "TR", "BR", "BL")):
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(vis, lbl, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(vis, "IPM source quad (warp these 4 floor points)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
class IntersectionDetection:

    def __init__(self, params: dict | None = None):
        self.params = dict(_DEFAULT_PARAMS)
        if params:
            self.params.update(params)

    @staticmethod
    def _orientation(box: np.ndarray) -> tuple[float, float, float]:
        e0 = box[1] - box[0]
        e1 = box[2] - box[1]
        n0 = float(np.linalg.norm(e0))
        n1 = float(np.linalg.norm(e1))
        if n0 >= n1:
            long_vec, length, width = e0, n0, n1
        else:
            long_vec, length, width = e1, n1, n0
        angle = float(np.degrees(np.arctan2(long_vec[1], long_vec[0]))) % 180.0
        return length, max(width, 1.0), angle

    # ---- stage 1: produce the work image (warped or ROI) --------------------
    def _work_image(self, frame_bgr: np.ndarray):
        """Returns (canvas_bgr, work_gray, oy, Pw, Ph).
        canvas = image to annotate; work_gray = image to threshold;
        oy = y offset from work coords -> canvas coords;
        Pw, Ph = work image dimensions (zone reference)."""
        p = self.params
        H, W = frame_bgr.shape[:2]
        if int(p["ipm_enable"]):
            Ww = max(60, int(p["warp_w"]))
            Wh = max(60, int(p["warp_h"]))
            src = src_quad(frame_bgr.shape, p)
            dst = np.float32([[0, 0], [Ww, 0], [Ww, Wh], [0, Wh]])
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(frame_bgr, M, (Ww, Wh))
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            return warped, gray, 0, Ww, Wh
        # fallback: ROI crop on raw frame
        roi_y = max(0, min(int(H * float(p["roi_top"])), H - 2))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return frame_bgr.copy(), gray[roi_y:, :], roi_y, W, (H - roi_y)

    # ---- stage 3: segments --------------------------------------------------
    def _segments(self, binary: np.ndarray, oy: int) -> list[dict]:
        p = self.params
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < float(p["seg_min_area"]) or area > float(p["seg_max_area"]):
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)            # work coords
            length, width, angle = self._orientation(box)
            if length / width < float(p["seg_min_aspect"]):
                continue
            (cx, cy), _, _ = rect
            segs.append({
                "cx": float(cx), "cy": float(cy),
                "angle": angle, "length": length,
                "box_work": box,
                "box_canvas": (box + np.array([0, oy])).astype(int),
                "status": "loose",
            })
        return segs

    # ---- stage 4: greedy collinear grouping ---------------------------------
    def _group(self, segs: list[dict]) -> list[list[int]]:
        p = self.params
        atol = float(p["group_angle_tol"])
        ptol = float(p["group_perp_tol"])
        order = sorted(range(len(segs)), key=lambda i: -segs[i]["length"])
        used = [False] * len(segs)
        groups: list[list[int]] = []
        for i in order:
            if used[i]:
                continue
            seed = segs[i]
            used[i] = True
            members = [i]
            a = np.radians(seed["angle"])
            dirx, diry = np.cos(a), np.sin(a)
            nx, ny = -diry, dirx
            px, py = seed["cx"], seed["cy"]
            for j in order:
                if used[j]:
                    continue
                s = segs[j]
                if _angle_diff(s["angle"], seed["angle"]) > atol:
                    continue
                perp = abs((s["cx"] - px) * nx + (s["cy"] - py) * ny)
                if perp > ptol:
                    continue
                used[j] = True
                members.append(j)
            groups.append(members)
        return groups

    # ---- stage 5: classify a group as a dashed line -------------------------
    def _classify(self, members: list[int], segs: list[dict]) -> dict:
        p = self.params
        n = len(members)
        info = {"dashed": False, "n": n, "fill": 0.0, "resid": 0.0,
                "angle": 0.0, "pa": None, "pb": None}
        if n < int(p["min_dashes_in_line"]):
            return info

        pts = np.array([[segs[m]["cx"], segs[m]["cy"]] for m in members],
                       dtype=np.float32)
        vx, vy, x0, y0 = (float(v) for v in
                          cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel())
        base = np.array([x0, y0])
        dirv = np.array([vx, vy])
        nrm = np.array([-vy, vx])
        resid = float(np.max(np.abs((pts - base) @ nrm)))

        a_all, b_all, covered = [], [], 0.0
        for m in members:
            proj = (segs[m]["box_work"] - base) @ dirv
            a, b = float(proj.min()), float(proj.max())
            covered += (b - a)
            a_all.append(a)
            b_all.append(b)
        tmin, tmax = min(a_all), max(b_all)
        span = max(tmax - tmin, 1.0)
        fill = min(covered / span, 1.0)

        angle = float(np.degrees(np.arctan2(vy, vx))) % 180.0
        info.update({
            "fill": fill, "resid": resid, "angle": angle,
            "pa": base + tmin * dirv, "pb": base + tmax * dirv,
        })
        info["dashed"] = (
            float(p["min_fill_ratio"]) <= fill <= float(p["max_fill_ratio"])
            and resid <= float(p["max_resid_px"])
        )
        return info

    # ---- stage 6: arm assignment (in work coords) ---------------------------
    def _arms_of_line(self, info: dict, Pw: int, Ph: int) -> set[str]:
        p = self.params
        arms: set[str] = set()
        pa, pb = info["pa"], info["pb"]
        xs = (pa[0], pb[0])
        ys = (pa[1], pb[1])
        midx, midy = float(np.mean(xs)), float(np.mean(ys))
        lx, rx = float(p["left_edge"]) * Pw, float(p["right_edge"]) * Pw
        fy, by = float(p["front_edge"]) * Ph, float(p["back_edge"]) * Ph

        d_to_horiz = min(info["angle"], 180.0 - info["angle"])
        horizontal = d_to_horiz <= float(p["horiz_vert_split"])

        if horizontal:                      # crossing line -> lateral road
            if min(xs) < lx:
                arms.add("left")
            if max(xs) > rx:
                arms.add("right")
        else:                               # runs forward/back -> branch lane
            if midx < lx:
                arms.add("left")
            elif midx > rx:
                arms.add("right")
            else:
                if midy < fy:
                    arms.add("front")
                elif midy > by:
                    arms.add("back")
        return arms

    # ---- top level ----------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> dict:
        p = self.params
        out: dict = {
            "detected": False,
            "arms":     {a: False for a in ARM_ORDER},
            "counts":   {a: 0 for a in ARM_ORDER},
            "segments": [], "lines": [],
            "canvas": None, "binary": None,
            "oy": 0, "pw": 0, "ph": 0,
            "ipm": bool(int(p["ipm_enable"])),
        }
        if frame_bgr is None or frame_bgr.size == 0:
            return out

        canvas, work_gray, oy, Pw, Ph = self._work_image(frame_bgr)
        out["canvas"] = canvas
        out["oy"], out["pw"], out["ph"] = oy, Pw, Ph

        k = int(p["blur"]) | 1
        if k >= 3:
            work_gray = cv2.GaussianBlur(work_gray, (k, k), 0)
        block = max(3, int(p["adapt_block"]) | 1)
        binary = cv2.adaptiveThreshold(
            work_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, float(p["adapt_c"]))
        mk = int(p["morph"])
        if mk >= 2:
            kernel = np.ones((mk, mk), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        out["binary"] = binary

        segs = self._segments(binary, oy)
        groups = self._group(segs)

        vote = {a: 0 for a in ARM_ORDER}
        for members in groups:
            info = self._classify(members, segs)
            if not info["dashed"]:
                for m in members:
                    segs[m]["status"] = "rejected"
                continue
            line_arms = self._arms_of_line(info, Pw, Ph)
            for a in line_arms:
                vote[a] += 1
            for m in members:
                segs[m]["status"] = "accepted"
            out["lines"].append({
                "pa": (int(info["pa"][0]), int(info["pa"][1] + oy)),
                "pb": (int(info["pb"][0]), int(info["pb"][1] + oy)),
                "arms": sorted(line_arms),
                "fill": info["fill"], "resid": info["resid"], "n": info["n"],
            })

        thr = int(p["min_lines_per_arm"])
        for a in ARM_ORDER:
            out["counts"][a] = vote[a]
            out["arms"][a] = vote[a] >= thr
        out["detected"] = any(out["arms"].values())
        out["segments"] = [{"box": s["box_canvas"], "status": s["status"]}
                           for s in segs]
        return out


def arms_to_str(arms: dict) -> str:
    present = [a for a in ARM_ORDER if arms.get(a)]
    return ",".join(present) if present else "none"


def draw_overlay(frame: np.ndarray, r: dict, params: dict) -> np.ndarray:
    canvas = r.get("canvas")
    vis = (canvas if canvas is not None else frame).copy()
    oy, Pw, Ph = r["oy"], r["pw"], r["ph"]

    lx = int(float(params["left_edge"]) * Pw)
    rx = int(float(params["right_edge"]) * Pw)
    fy = oy + int(float(params["front_edge"]) * Ph)
    by = oy + int(float(params["back_edge"]) * Ph)
    cv2.line(vis, (0, oy), (vis.shape[1], oy), (255, 200, 0), 1)
    cv2.line(vis, (lx, oy), (lx, oy + Ph), (120, 120, 120), 1)
    cv2.line(vis, (rx, oy), (rx, oy + Ph), (120, 120, 120), 1)
    cv2.line(vis, (lx, fy), (rx, fy), (120, 120, 120), 1)
    cv2.line(vis, (lx, by), (rx, by), (120, 120, 120), 1)

    for s in r["segments"]:
        if s["status"] == "accepted":
            color = (0, 255, 0)
        elif s["status"] == "rejected":
            color = (0, 0, 230)
        else:
            color = (110, 110, 110)
        cv2.drawContours(vis, [s["box"]], 0, color, 2)

    for ln in r["lines"]:
        cv2.line(vis, ln["pa"], ln["pb"], (0, 255, 0), 2)
        label = "".join(a[0].upper() for a in ln["arms"]) or "-"
        mid = ((ln["pa"][0] + ln["pb"][0]) // 2, (ln["pa"][1] + ln["pb"][1]) // 2)
        cv2.putText(vis, f"{label} f{ln['fill']:.2f}", (mid[0] - 18, mid[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cnt = r["counts"]
    tag = arms_to_str(r["arms"]).upper()
    mode = "IPM" if r.get("ipm") else "ROI"
    hud = (f"[{mode}] ARMS: {tag}   F{cnt['front']} B{cnt['back']} "
           f"L{cnt['left']} R{cnt['right']}   lines={len(r['lines'])}")
    color = (0, 255, 0) if r["detected"] else (0, 165, 255)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (40, 40, 40), -1)
    cv2.putText(vis, hud, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
class IntersectionDetectorNode(Node):

    def __init__(self):
        super().__init__("intersection_detector")

        self.declare_parameter(
            "image_topic", "/camera/image_raw",
            ParameterDescriptor(description="Input image topic"))
        self.declare_parameter(
            "params_config", "",
            ParameterDescriptor(description="Optional path to intersection_params.yaml"))

        for k, v in _DEFAULT_PARAMS.items():
            self.declare_parameter(
                k, float(v) if isinstance(v, float) else int(v),
                ParameterDescriptor(description=f"Intersection param: {k}"))

        yaml_path = self.get_parameter("params_config").value
        yaml_params = _load_params_yaml(yaml_path)
        if yaml_params:
            self.get_logger().info(f"Params loaded from: {yaml_path}")
            for k, v in yaml_params.items():
                ptype = (rclpy.parameter.Parameter.Type.DOUBLE
                         if isinstance(v, float)
                         else rclpy.parameter.Parameter.Type.INTEGER)
                self.set_parameters(
                    [rclpy.parameter.Parameter(k, ptype, v)])
        else:
            self.get_logger().info("Using default params (no YAML)")

        self.bridge   = CvBridge()
        self.detector = IntersectionDetection(self._snapshot_params())

        self._pending = "none"
        self._pending_n = 0
        self._confirmed = "none"

        image_topic = self.get_parameter("image_topic").value
        self.sub_img      = self.create_subscription(
            Image, image_topic, self._on_image, 10)
        self.pub_detected = self.create_publisher(Bool,   "/intersection/detected", 10)
        self.pub_arms     = self.create_publisher(String, "/intersection/arms",     10)
        self.pub_debug    = self.create_publisher(Image,  "/vision/intersection",   10)

        self.get_logger().info(
            f"IntersectionDetectorNode (v3 IPM) ready | topic={image_topic} | "
            f"ipm={'on' if int(self.detector.params['ipm_enable']) else 'off'} | "
            f"warp={int(self.detector.params['warp_w'])}x{int(self.detector.params['warp_h'])}")

    def _snapshot_params(self) -> dict:
        return {k: self.get_parameter(k).value for k in _DEFAULT_PARAMS}

    def _on_image(self, msg: Image):
        self.detector.params.update(self._snapshot_params())
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        r = self.detector.detect(frame)
        raw = arms_to_str(r["arms"])

        n_need = int(self.detector.params["debounce"])
        if raw == self._pending:
            self._pending_n += 1
        else:
            self._pending = raw
            self._pending_n = 1
        if self._pending_n >= n_need and self._pending != self._confirmed:
            self._confirmed = self._pending
            self.get_logger().info(f"INTERSECTION: {self._confirmed.upper()}")

        d_msg = Bool();   d_msg.data = (self._confirmed != "none")
        a_msg = String(); a_msg.data = self._confirmed
        self.pub_detected.publish(d_msg)
        self.pub_arms.publish(a_msg)

        if self.pub_debug.get_subscription_count() > 0 and r["canvas"] is not None:
            vis = draw_overlay(frame, r, self.detector.params)
            out_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            out_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_debug.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = IntersectionDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()