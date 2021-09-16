# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2roi
from ..builder import HEADS, build_head
from .cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class CascadeMaskScoringRoIHead(CascadeRoIHead):
    """Mask Scoring RoIHead for Mask Scoring RCNN.

    https://arxiv.org/abs/1903.00241
    """

    def __init__(self, mask_iou_head, **kwargs):
        assert mask_iou_head is not None
        super(CascadeMaskScoringRoIHead, self).__init__(**kwargs)
        self.mask_iou_head = build_head(mask_iou_head)

    def _mask_forward_train(self, stage, x, sampling_results, gt_masks,
                            rcnn_train_cfg, img_metas, bbox_feats = None):
        """Run forward function and calculate loss for Mask head in
        training."""
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        mask_results = super(CascadeMaskScoringRoIHead,
                             self)._mask_forward_train(stage, x, sampling_results,
                                                       gt_masks, rcnn_train_cfg,
                                                       img_metas, bbox_feats = bbox_feats)

        if mask_results['loss_mask'] is None:
            return mask_results

        # mask iou head forward and loss
        pos_mask_pred = mask_results['mask_pred'][
            range(mask_results['mask_pred'].size(0)), pos_labels]
        mask_iou_pred = self.mask_iou_head(mask_results['mask_feats'],
                                           pos_mask_pred)
        pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)),
                                          pos_labels]

        mask_iou_targets = self.mask_iou_head.get_targets(
            stage, sampling_results, gt_masks, pos_mask_pred,
            mask_results['mask_targets'], self.train_cfg)
        loss_mask_iou = self.mask_iou_head.loss(pos_mask_iou_pred,
                                                mask_iou_targets)
        mask_results['loss_mask'].update(loss_mask_iou)
        return mask_results

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Obtain mask prediction without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            num_classes = self.mask_head.num_classes
            segm_results = [[[] for _ in range(num_classes)]
                            for _ in range(num_imgs)]
            mask_scores = [[[] for _ in range(num_classes)]
                           for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i]
                for i in range(num_imgs)
            ]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            concat_det_labels = torch.cat(det_labels)
            # get mask scores with mask iou head
            mask_feats = mask_results['mask_feats']
            mask_pred = mask_results['mask_pred']
            mask_iou_pred = self.mask_iou_head(
                mask_feats, mask_pred[range(concat_det_labels.size(0)),
                                      concat_det_labels])
            # split batch mask prediction back to each image
            num_bboxes_per_img = tuple(len(_bbox) for _bbox in _bboxes)
            mask_preds = mask_pred.split(num_bboxes_per_img, 0)
            mask_iou_preds = mask_iou_pred.split(num_bboxes_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            mask_scores = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                    mask_scores.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], _bboxes[i], det_labels[i],
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    # get mask scores with mask iou head
                    mask_score = self.mask_iou_head.get_mask_scores(
                        mask_iou_preds[i], det_bboxes[i], det_labels[i])
                    segm_results.append(segm_result)
                    mask_scores.append(mask_score)
        return list(zip(segm_results, mask_scores))
