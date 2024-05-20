# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on EVA: Exploring the Limits of Masked Visual Representation Learning at Scale (https://arxiv.org/abs/2211.07636)
# https://github.com/baaivision/EVA/tree/master/EVA-01
# --------------------------------------------------------'


import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma, AverageMeter

import utils


def cul_muti_acc_f1(output, targets, th=0.1):
    
    # Convert to boolean based on threshold
    prediction = (output > th).bool()
    target_bool = targets.bool()

    # Calculate True Positives, False Positives, False Negatives
    true_positives = (prediction & target_bool).float().sum()
    false_positives = (prediction & ~target_bool).float().sum()
    false_negatives = (~prediction & target_bool).float().sum()

    # Precision, Recall, and F1 Score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

from PIL import Image, ImageDraw, ImageFont

def add_text_to_image(image_path, predict_class_name, gd_class_name, save_path):
    # 打开图像文件
    img = Image.open(image_path)
    original_width, original_height = img.size

    # 设置新图像宽度和字体
    additional_width = original_width*0.3  # 可以根据需要调整空白区域的宽度
    new_width = original_width + int(additional_width)
    font_size = 30  # 字体大小
    try:
        # 尝试加载一个好看的字体，如果有自定义字体可以在这里指定
        font = ImageFont.truetype("/data/jiahaoguo/QwenWQwl2/tools/方正粗黑宋简体.TTF", size=font_size)
    except IOError:
        # 如果无法加载字体文件，使用默认字体
        font = ImageFont.load_default()

    # 创建一个新图像，宽度更宽，背景为白色
    new_img = Image.new('RGB', (new_width, original_height), 'white')
    new_img.paste(img, (0, 0))

    # 在新图像上绘制文本
    draw = ImageDraw.Draw(new_img)
    text_y_position = 10  # 开始绘制文本的y坐标

    # 写入预测的类名
    draw.text((original_width + 5, text_y_position), "Predicted Classes:", fill="black", font=font)
    text_y_position += font_size  # 调整文本位置
    for name in predict_class_name:
        draw.text((original_width + 5, text_y_position), name, fill="black", font=font)
        text_y_position += font_size

    # 写入实际的类名
    text_y_position += font_size  # 添加额外的空间
    draw.text((original_width + 5, text_y_position), "Ground Truth Classes:", fill="black", font=font)
    text_y_position += font_size  # 调整文本位置
    for name in gd_class_name:
        draw.text((original_width + 5, text_y_position), name, fill="black", font=font)
        text_y_position += font_size

    # 保存新图像
    new_img.save(save_path)

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs.float(), target.float())
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale, optimizer._global_grad_norm


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    # Set the logging level to WARNING to reduce output (only warnings and errors will be displayed)
    
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # if data_iter_step > 20: break

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if args.linear_probe:
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it]
                else:
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.bfloat16() if args.bf16 else samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value, grad_norm = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if args.muti_lables:
            _,_,class_acc = cul_muti_acc_f1(output, targets)
        elif mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

import numpy as np
import os,json

def update_f1_acc(results, result, thresh):
    acc, recall, f1 = result
    print(f"thresh {thresh}: acc:{acc}, recall:{recall}, f1:{f1}")
    results[f"acc({thresh})"] = acc
    results[f"recall({thresh})"] = recall
    results[f"f1({thresh})"] = f1
    return results


@torch.no_grad()
def evaluate(data_loader, model, device, muti_lable=False, ds=False, bf16=False, args=None):
    if muti_lable:
        criterion = torch.nn.MultiLabelSoftMarginLoss(reduction='sum')
        print("now muti lable test")
        output_all = []
        target_all = []
    else:
        criterion = torch.nn.CrossEntropyLoss() # linear
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        if ds:
            images = images.bfloat16() if bf16 else images.half()
            output = model(images)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)   

        if muti_lable:
            output = torch.sigmoid(output)

        if args.add_mode in ["plot", "save_pred"]:
            prediction = (output > 0.4).bool()
            target_bool = target.bool()
            paths = batch[-1]
            os.makedirs(args.add_mode_dir, exist_ok=True)
            for i in range(prediction.shape[0]):
                true_indices = torch.where(prediction[i])[0]
                if len(true_indices)==0:
                    continue

                true_class_code = [data_loader.dataset.classes[index] for index in true_indices]
                predict_class_name = [data_loader.dataset.code_name_map[code] for code in true_class_code]
                # predict_class_name = [f"{name} ({output[i][index]:.3f})" for name, index in zip(predict_class_name, true_indices)]
                gd_indices = torch.where(target_bool[i])[0]
                gt_class_code = [data_loader.dataset.classes[index] for index in gd_indices]
                gd_class_name = [data_loader.dataset.code_name_map[code] for code in gt_class_code]
                save_path = os.path.join(args.add_mode_dir, os.path.basename(paths[i]))
                if args.add_mode == "plot":
                    add_text_to_image(paths[i], predict_class_name, gd_class_name, save_path)
                

                save_txt_dict = {
                    "model_new_label":[],
                    "makeLabels":[],
                    "imagePath": os.path.abspath(paths[i])
                }
                merge_code = list(true_class_code)
                merge_code.extend(gt_class_code)
                merge_code = set(merge_code)
                merge_name = [data_loader.dataset.code_name_map[code] for code in merge_code]
                for j, (label_pre,code_pre) in enumerate(zip(merge_name, merge_code)):
                    if code_pre in gt_class_code and code_pre in true_class_code: #一致
                        tip = " x"
                    elif code_pre not in gt_class_code and code_pre in true_class_code:#人没标， 模型预测
                        tip = " +"
                    elif code_pre in gt_class_code and code_pre not in true_class_code:#人标了， 模型没预测
                        tip = " -"

                    save_txt_dict["model_new_label"].append({
                        "code": code_pre, "name": label_pre+tip, "index": j
                    })

                for j, (label_pre,code_pre) in enumerate(zip(gd_class_name, gt_class_code)):
                    save_txt_dict["makeLabels"].append({
                        "code": code_pre, "name": label_pre, "index": j
                    })

                # 打开文件进行写入
                # import pdb;pdb.set_trace()
                if args.add_mode == "save_pred":
                    path_ = "/".join((".".join(paths[i].split(".")[:-1])+"_pred.txt").split("/")[-3:])
                    save_dir = args.add_mode_dir
                    save_path = os.path.join(save_dir, path_)
                    os.makedirs(os.path.dirname(save_path),exist_ok=True)
                    # print(os.path.join(save_dir, path_))
                    with open(save_path, 'w', encoding='utf-8') as f:
                        # 使用 json.dump 方法将字典保存为 JSON 格式的字符串，并设置 ensure_ascii=False 以支持中文字符
                        json.dump(save_txt_dict, f, ensure_ascii=False, indent=4)


        if muti_lable:
            output_all.append(output)
            target_all.append(target)
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if muti_lable:
        results = {}
        # import pdb;pdb.set_trace()
        output_all = torch.cat(output_all)
        target_all = torch.cat(target_all)
        for i in range(5,10):
            results = update_f1_acc(results, cul_muti_acc_f1(output_all, target_all, th=0.1*i), 0.1*i)
        
        results["mean_acc(0.5-0.9)"] = f"{np.array([v.cpu() for key, v in results.items() if 'acc' in key]).mean():.3f}"
        results["mean_rec(0.5-0.9)"] = f"{np.array([v.cpu() for key, v in results.items() if 'recall' in key]).mean():.3f}"
        results["mean_f1(0.5-0.9)"] = f"{np.array([v.cpu() for key, v in results.items() if 'f1' in key]).mean():.3f}"

        return {k: meter for k, meter in results.items() if "mean" in k}
    else:
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_real(data_loader, model, device, real_labels, ds=False, bf16=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if ds:
            images = images.bfloat16() if bf16 else images.half()
            output = model(images)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)       

        if real_labels is not None:
            real_labels.add_result(output)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)

    print('* ReaL Acc@1 {:.3f} Acc@5 {:.3f} loss {losses.global_avg:.3f}'
          .format(top1a, top5a, losses=metric_logger.loss))

    exit(0)



