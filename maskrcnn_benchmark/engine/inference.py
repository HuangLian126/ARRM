# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

def compute_on_dataset(oneStage, model, data_loader, data_loader_sup, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    data_loader_sup = iter(data_loader_sup)
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                sups, supTarget = next(data_loader_sup)
                output = model(images.to(device), [target.to(device) for target in targets], [sup.to(device) for sup in sups], supTarget.to(device), oneStage)
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update({img_id: result for img_id, result in zip(image_ids, output)})
    return results_dict

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning("Number of images that were gathered from multiple processes is not a contiguous set. Some images might be missing from the evaluation.")
    predictions = [predictions[i] for i in image_ids]
    return predictions

def inference(oneStage, model, data_loader, data_loader_sup, dataset_name, iou_types=("bbox",), box_only=False,
              device="cuda", expected_results=(), expected_results_sigma_tol=4, output_folder=None):
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset

    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(oneStage, model, data_loader, data_loader_sup, device, inference_timer)

    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info("Total run time: {} ({} s / img per device, on {} devices)".format(total_time_str, total_time * num_devices / len(dataset), num_devices))
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info("Model inference time: {} ({} s / img per device, on {} devices)".format(total_infer_time, inference_timer.total_time * num_devices / len(dataset), num_devices))

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(box_only=box_only, iou_types=iou_types, expected_results=expected_results, expected_results_sigma_tol=expected_results_sigma_tol)

    return evaluate(dataset=dataset, predictions=predictions, output_folder=output_folder, **extra_args)