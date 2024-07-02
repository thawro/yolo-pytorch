import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm.auto import tqdm

from src.annots.ops import xyxy2xywh_coco
from src.base.validator import BaseValidator
from src.datasets.coco.utils import coco80_to_coco91_labels
from src.detection.results import DetectionResult
from src.logger.pylogger import log_msg


class CocoDetectionValidator(BaseValidator):
    results: list[DetectionResult]

    def __init__(self, gt_annot_filepath: str, preds_filepath: str):
        super().__init__()
        self.coco_gt = COCO(gt_annot_filepath)
        self.gt_annot_filepath = gt_annot_filepath
        self.preds_filepath = preds_filepath

    def evaluate(self) -> dict[str, float]:
        self.process_results()
        coco_results = []
        image_ids = []
        for result in self.results:
            img_file_stem = Path(result.image_filepath).stem
            image_id = int(img_file_stem) if img_file_stem.isnumeric() else img_file_stem
            image_ids.append(image_id)
            pd_boxes_xywh = xyxy2xywh_coco(result.pd_boxes).tolist()
            pd_classes = result.pd_classes.tolist()
            pd_conf = result.pd_conf.tolist()
            for i in range(result.num_objects):
                coco_eval_dict = {
                    "image_id": image_id,
                    "category_id": coco80_to_coco91_labels[int(pd_classes[i])],
                    "bbox": [round(coord, 3) for coord in pd_boxes_xywh[i]],
                    "score": round(pd_conf[i], 5),
                }
                coco_results.append(coco_eval_dict)
        with open(str(self.preds_filepath), "w") as f:
            json.dump(coco_results, f)
            log_msg(f"{self.prefix}Saved prediction annotations to {self.preds_filepath}")
        coco_pd = self.coco_gt.loadRes(str(self.preds_filepath))
        coco_evaluator = COCOeval(self.coco_gt, coco_pd, "bbox")

        coco_evaluator.params.imgIds = image_ids
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        AP_50_95_all, AP_50_all = coco_evaluator.stats[:2]
        return {"mAP_50": AP_50_all, "mAP_50-95": AP_50_95_all}
