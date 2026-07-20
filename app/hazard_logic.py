"""
Post-detection hazard logic.
Classes: worker (0), dangerous_vehicle (1).

Three hazard use cases, all computed on top of raw detections:
  - Proximity: a worker box and a vehicle box are close enough
    (by normalized center distance or box overlap) to count as a risk.
  - Vehicle movement: a vehicle centroid displacing fast enough across
    detection passes while workers are on site (moving machinery caution).
  - Height (EXPERIMENTAL heuristic): a worker whose box sits in the upper
    zone of the frame — a stand-in until height-labeled/synthetic data
    exists; off by default (config.HEIGHT_ZONE_ENABLED).
"""

from dataclasses import dataclass


@dataclass
class Detection:
    cls: int
    conf: float
    xyxy: tuple  # x1, y1, x2, y2 in pixel coords
    flags: tuple = ()  # optional extra tags, e.g. ("FALLEN?",) from the pose overlay


WORKER = 0
VEHICLE = 1


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def boxes_overlap(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def find_proximity_hazards(detections: list[Detection], frame_diag: float, distance_ratio: float = 0.25):
    """
    Return list of (worker_det, vehicle_det) pairs flagged as proximity hazards.
    distance_ratio: hazard threshold as a fraction of the frame diagonal.
    """
    workers = [d for d in detections if d.cls == WORKER]
    vehicles = [d for d in detections if d.cls == VEHICLE]

    threshold = frame_diag * distance_ratio
    hazards = []
    for w in workers:
        for v in vehicles:
            if boxes_overlap(w.xyxy, v.xyxy) or box_distance(w.xyxy, v.xyxy) < threshold:
                hazards.append((w, v))
    return hazards


def find_height_hazards(detections: list[Detection], frame_height: float,
                        zone_fraction: float = 0.45, use_zone: bool = True,
                        relative_gap: float = 0.30):
    """EXPERIMENTAL height heuristics (a stand-in until synthetic
    height-labeled data exists):

    - Relative (always applied, needs 2+ workers): a worker whose feet are
      more than relative_gap of the frame height above the lowest worker's
      feet is treated as elevated. Robust to camera tilt because it compares
      workers against each other, not against the frame.
    - Absolute zone (use_zone): feet above zone_fraction of the frame height.
      Only sensible with a level, ground-covering camera view.
    """
    workers = [d for d in detections if d.cls == WORKER]
    flagged: list[Detection] = []
    if len(workers) >= 2:
        lowest_feet = max(w.xyxy[3] for w in workers)
        gap = frame_height * relative_gap
        flagged = [w for w in workers if lowest_feet - w.xyxy[3] > gap]
    if use_zone:
        limit = frame_height * zone_fraction
        flagged += [w for w in workers
                    if w.xyxy[3] < limit and w not in flagged]
    return flagged


PPE_VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest"}


def _correlates_with_worker(worker_box, ppe_box, x_overlap_min: float = 0.4,
                            y_margin_ratio: float = 1.0) -> bool:
    """PPE_Detect and the worker detector are different models trained on
    different data — their boxes for the same person do not line up
    pixel-for-pixel. Measured on real footage: a NO-Safety Vest box can sit
    ~0.6x a worker-box-height entirely below the worker box (the worker
    detector's box ends around the torso; the vest region reads lower), so
    literal rectangle intersection misses genuine matches. This instead
    requires solid horizontal overlap (same person, side to side) plus a
    generous vertical margin below the worker box to account for that gap.
    """
    wx1, wy1, wx2, wy2 = worker_box
    px1, py1, px2, py2 = ppe_box
    x_overlap = max(0, min(wx2, px2) - max(wx1, px1))
    x_span_min = min(wx2 - wx1, px2 - px1)
    if x_span_min <= 0 or x_overlap / x_span_min < x_overlap_min:
        return False
    h = wy2 - wy1
    return py1 <= wy2 + h * y_margin_ratio and py2 >= wy1 - h * 0.2


def find_ppe_violations(workers: list[Detection], ppe_detections: list[dict]) -> list[Detection]:
    """Baseline PPE compliance: worker Detections correlated with an
    explicit PPE_Detect negative-class box (NO-Hardhat / NO-Safety Vest).

    Absence of any PPE detection on a worker is deliberately NOT treated as
    a violation — only a positive negative-class hit counts. A worker the
    PPE model simply didn't see (angle, distance, occlusion) would otherwise
    flag constantly and make the baseline check noise instead of signal.
    """
    negatives = [d["xyxy"] for d in ppe_detections if d["name"] in PPE_VIOLATION_CLASSES]
    return [w for w in workers
           if any(_correlates_with_worker(w.xyxy, n) for n in negatives)]


class VehicleMotionTracker:
    """Tracks vehicle centroids across detection passes and flags vehicles
    moving faster than move_ratio_per_sec (fraction of frame diagonal per
    second). Association is nearest-centroid — adequate for the few large
    machines a site camera sees at once."""

    def __init__(self, frame_diag: float, move_ratio_per_sec: float = 0.04):
        self.frame_diag = frame_diag
        self.threshold_per_sec = frame_diag * move_ratio_per_sec
        self._last: list[tuple[float, float]] = []
        self._last_time: float | None = None

    def update(self, detections: list[Detection], now: float) -> list[Detection]:
        """Returns the vehicles considered to be in motion this pass."""
        vehicles = [d for d in detections if d.cls == VEHICLE]
        centers = [box_center(d.xyxy) for d in vehicles]
        moving = []
        if self._last and self._last_time is not None:
            dt = max(now - self._last_time, 1e-3)
            # cap dt so a long stall doesn't make everything look stationary
            dt = min(dt, 2.0)
            for det, (cx, cy) in zip(vehicles, centers):
                dist = min(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                           for px, py in self._last)
                # ignore jumps larger than 1/4 diagonal: that's a new vehicle
                # or an association error, not motion
                if self.threshold_per_sec * dt < dist < self.frame_diag * 0.25:
                    moving.append(det)
        self._last = centers
        self._last_time = now
        return moving
