"""
Pluggable detector backends so different approaches can be swapped and
compared in the app without touching the video pipeline.

Every detector implements predict_frame(frame, conf, imgsz) -> list[Detection]
with classes worker (0) / dangerous_vehicle (1). To add a new approach:
subclass BaseDetector and add one entry to DETECTOR_REGISTRY.
"""

from pathlib import Path

import config
from hazard_logic import Detection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAN_A_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "plan_a_yolov8" / "weights" / "best.pt"

WORKER_PROMPTS = ["person", "worker", "construction worker"]
VEHICLE_PROMPTS = ["truck", "excavator", "crane", "bulldozer", "forklift", "vehicle", "loader"]


def _boxes_to_detections(result, class_of=None) -> list[Detection]:
    dets = []
    if result.boxes is None:
        return dets
    for box in result.boxes:
        raw_cls = int(box.cls.item())
        cls = class_of(raw_cls) if class_of else raw_cls
        if cls is None:
            continue
        conf = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        dets.append(Detection(cls=cls, conf=conf, xyxy=(x1, y1, x2, y2)))
    return dets


class BaseDetector:
    key = "base"
    label = "Base"
    # Box detectors return Detection lists; summary detectors (supports_boxes=False)
    # return per-frame dicts via summarize_frame instead — no spatial output.
    supports_boxes = True

    def __init__(self, weights_path: str | None = None, device: str = "0"):
        self.weights_path = str(weights_path or PLAN_A_WEIGHTS)
        self.device = device

    def load(self):
        raise NotImplementedError

    def predict_frame(self, frame, conf: float, imgsz: int) -> list[Detection]:
        raise NotImplementedError

    def summarize_frame(self, frame, conf: float, imgsz: int) -> dict:
        """For supports_boxes=False detectors: return
        {"workers": int, "vehicles": int, "hazard": bool, "note": str}."""
        raise NotImplementedError


class PlanADetector(BaseDetector):
    key = "plan_a"
    label = "Plan A — Fine-tuned YOLOv8"

    def __init__(self, weights_path=None, device="0", model=None):
        super().__init__(weights_path, device)
        self.model = model

    def load(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
        return self

    def predict_frame(self, frame, conf, imgsz):
        quantize = "fp16" if self.device != "cpu" else None
        result = self.model.predict(
            frame, imgsz=imgsz, conf=conf, device=self.device, quantize=quantize, verbose=False
        )[0]
        return _boxes_to_detections(result)


class PlanBDetector(BaseDetector):
    key = "plan_b"
    label = "Plan B — YOLO-World zero-shot"

    def __init__(self, weights_path=None, device="0", model=None):
        super().__init__(weights_path, device)
        self.model = model
        prompts = WORKER_PROMPTS + VEHICLE_PROMPTS
        self._prompt_class = {i: (0 if p in WORKER_PROMPTS else 1) for i, p in enumerate(prompts)}
        self._prompts = prompts

    def load(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO("yolov8s-world.pt")
            self.model.set_classes(self._prompts)
        return self

    def predict_frame(self, frame, conf, imgsz):
        result = self.model.predict(
            frame, imgsz=imgsz, conf=conf, device=self.device, verbose=False
        )[0]
        return _boxes_to_detections(result, class_of=self._prompt_class.get)


class PlanCDetector(PlanADetector):
    """Plan A detections plus a YOLOv8-pose pass that tags workers whose
    posture looks fallen/abnormal (box wider than tall, or shoulders not
    above hips)."""

    key = "plan_c"
    label = "Plan C — YOLOv8 + pose overlay"

    L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

    def __init__(self, weights_path=None, device="0", model=None, pose_model=None):
        super().__init__(weights_path, device, model)
        self.pose_model = pose_model

    def load(self):
        super().load()
        if self.pose_model is None:
            from ultralytics import YOLO
            self.pose_model = YOLO("yolov8s-pose.pt")
        return self

    def _abnormal(self, kpts, box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if h > 0 and w / h > 1.2:
            return True
        shoulder_y = [kpts[i][1] for i in (self.L_SHOULDER, self.R_SHOULDER) if kpts[i][2] > 0.3]
        hip_y = [kpts[i][1] for i in (self.L_HIP, self.R_HIP) if kpts[i][2] > 0.3]
        return bool(shoulder_y and hip_y and
                    sum(shoulder_y) / len(shoulder_y) >= sum(hip_y) / len(hip_y))

    def predict_frame(self, frame, conf, imgsz):
        dets = super().predict_frame(frame, conf, imgsz)
        pose_result = self.pose_model.predict(
            frame, imgsz=imgsz, conf=conf, device=self.device, verbose=False
        )[0]
        if pose_result.keypoints is None or pose_result.boxes is None:
            return dets
        for box, kpts in zip(pose_result.boxes, pose_result.keypoints.data):
            box_xyxy = [float(v) for v in box.xyxy[0]]
            if self._abnormal(kpts.tolist(), box_xyxy):
                for d in dets:
                    if d.cls == 0 and _iou(d.xyxy, box_xyxy) > 0.5:
                        d.flags = ("FALLEN?",)
        return dets


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0


class _GroundingDINO:
    """Shared text-to-box grounding module (Grounding DINO, arXiv:2303.05499).
    VLMs and anomaly models can describe or flag a hazard but cannot localise
    it; this converts text prompts into boxes for Plans D and E."""

    _instance = None
    PROMPT = "a worker. a truck. an excavator. a crane. a bulldozer. a forklift."

    def __init__(self, device: str = "0"):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = "IDEA-Research/grounding-dino-tiny"
        self.torch_device = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.torch_device)
        self._torch = torch

    @classmethod
    def get(cls, device: str = "0"):
        if cls._instance is None:
            cls._instance = cls(device)
        return cls._instance

    def ground(self, frame, conf: float) -> list[Detection]:
        import cv2
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, text=self.PROMPT, return_tensors="pt").to(self.torch_device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=max(conf, 0.25), text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = results.get("text_labels", results["labels"])
        dets = []
        for box, score, label in zip(results["boxes"], results["scores"], labels):
            cls = 0 if "worker" in str(label) else 1
            x1, y1, x2, y2 = [float(v) for v in box]
            dets.append(Detection(cls=cls, conf=float(score), xyxy=(x1, y1, x2, y2)))
        return dets


class PlanDDetector(BaseDetector):
    """Direct VLM for per-frame hazard reasoning, served by the Ollama
    server on the DGX Spark (config.VLM_MODELS chain, default
    qwen2.5vl:32b -> gemma3:27b -> llava:7b), with Grounding DINO supplying
    the bounding boxes the VLM cannot produce reliably itself (IoU < 20% in
    the literature). Roughly 1 s per sampled frame — use a high frame skip
    and short clips."""

    key = "plan_d"
    label = "Plan D — Direct VLM"
    supports_boxes = False

    PROMPT = (
        "Count the construction workers (people) and heavy/dangerous vehicles "
        "(trucks, excavators, cranes, bulldozers, forklifts) in this image, and say "
        "whether any worker is dangerously close to a vehicle. Respond ONLY with "
        'compact JSON: {"workers": <int>, "vehicles": <int>, "proximity_hazard": <true/false>}'
    )

    def load(self):
        from llm_client import get_client
        self._llm = get_client()
        self._grounder = _GroundingDINO.get(self.device)
        return self

    def summarize_frame(self, frame, conf, imgsz):
        import json
        import re

        import cv2

        detections = self._grounder.ground(frame, conf)

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return {"workers": 0, "vehicles": 0, "hazard": False,
                    "note": "encode failed", "detections": detections}
        text = self._llm.describe_image(self.PROMPT, buf.tobytes())
        if text:
            try:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                parsed = json.loads(match.group(0)) if match else {}
                model = self._llm.resolve(config.VLM_MODELS, need="vision") or "VLM"
                return {
                    "workers": int(parsed.get("workers", 0)),
                    "vehicles": int(parsed.get("vehicles", 0)),
                    "hazard": bool(parsed.get("proximity_hazard", False)),
                    "note": f"{model} + DINO boxes",
                    "detections": detections,
                }
            except Exception:
                pass
        return {"workers": 0, "vehicles": 0, "hazard": False,
                "note": "VLM unavailable, DINO boxes only", "detections": detections}


class PlanEDetector(BaseDetector):
    """Anomaly detection (ResNet18 features + IsolationForest). Calibrates on
    the first sampled frames of the uploaded video as 'normal', then flags
    frames that deviate. No object counts or boxes — offline eval showed this
    approach is near-random for proximity hazards; included for comparison."""

    key = "plan_e"
    label = "Plan E — Anomaly detection"
    supports_boxes = False

    WARMUP_FRAMES = 30

    def load(self):
        import torch
        from torchvision import models, transforms

        self._torch = torch
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = torch.nn.Identity()
        self._embed = resnet.eval()
        self._preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._buffer = []
        self._clf = None
        self._grounder = _GroundingDINO.get(self.device)
        return self

    def _features(self, frame):
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._torch.no_grad():
            return self._embed(self._preprocess(rgb).unsqueeze(0)).squeeze().numpy()

    def summarize_frame(self, frame, conf, imgsz):
        feat = self._features(frame)
        if self._clf is None:
            self._buffer.append(feat)
            if len(self._buffer) >= self.WARMUP_FRAMES:
                from sklearn.ensemble import IsolationForest
                import numpy as np
                self._clf = IsolationForest(random_state=42, contamination=0.1)
                self._clf.fit(np.stack(self._buffer))
            return {"workers": 0, "vehicles": 0, "hazard": False,
                    "note": f"calibrating {len(self._buffer)}/{self.WARMUP_FRAMES}"}
        score = float(-self._clf.score_samples([feat])[0])
        anomalous = bool(self._clf.predict([feat])[0] == -1)
        # Localise objects only on flagged frames — anomaly scoring itself has
        # no object concept, so grounding runs when there is something to show.
        detections = self._grounder.ground(frame, conf) if anomalous else []
        workers = sum(1 for d in detections if d.cls == 0)
        vehicles = sum(1 for d in detections if d.cls == 1)
        return {"workers": workers, "vehicles": vehicles, "hazard": anomalous,
                "note": f"anomaly score {score:.2f}" + (" + DINO boxes" if anomalous else ""),
                "detections": detections}


class PPEDetector:
    """Wraps PPE_Detect/best.pt (Roboflow-style Hardhat/Safety Vest/boots/
    gloves detector) used by live.py for the baseline PPE-compliance hazard
    and the height-alarm cross-check. Not a worker/vehicle Plan, so it stays
    out of DETECTOR_REGISTRY — live.py loads it directly by path."""

    def __init__(self, weights_path: str, device: str = "0"):
        self.weights_path = str(weights_path)
        self.device = device
        self.model = None
        self.names: dict[int, str] = {}

    def load(self):
        from ultralytics import YOLO
        self.model = YOLO(self.weights_path)
        self.names = self.model.names
        return self

    def predict(self, frame, conf: float, imgsz: int = 416) -> list[dict]:
        """Returns [{'name': str, 'conf': float, 'xyxy': (x1,y1,x2,y2)}, ...]
        using the model's own class names — never hardcode its class ids,
        they're specific to this weight file."""
        quantize = "fp16" if self.device != "cpu" else None
        result = self.model.predict(
            frame, imgsz=imgsz, conf=conf, device=self.device, quantize=quantize, verbose=False
        )[0]
        out = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                out.append({"name": self.names.get(cls_id, str(cls_id)),
                           "conf": float(box.conf.item()), "xyxy": (x1, y1, x2, y2)})
        return out


DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    PlanADetector.label: PlanADetector,
    PlanBDetector.label: PlanBDetector,
    PlanCDetector.label: PlanCDetector,
    PlanDDetector.label: PlanDDetector,
    PlanEDetector.label: PlanEDetector,
}


def create_detector(label: str, weights_path: str | None = None, device: str = "0") -> BaseDetector:
    cls = DETECTOR_REGISTRY[label]
    return cls(weights_path=weights_path, device=device).load()
