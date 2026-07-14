"""
Post-detection hazard logic (Plan A).
Classes: worker (0), dangerous_vehicle (1).

Proximity hazard: a worker box and a vehicle box are close enough
(by normalized center distance or box overlap) to count as a risk.
Height hazard is not implemented yet (no labeled data source, per project plan).
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
