# Copyright (c) Lê Hoàn Chân. All rights reserved.
from mmengine.structures import BaseDataElement, InstanceData, LabelData

class TableMasterDataSample(BaseDataElement):
    """A data structure for table master with two heads in models.

    The attributes in 'TableMasterDataSample' are divided into four parts:

        - 'gt_instances'(InstanceData): Ground truth of instance annotations (bboxes, masks, etc).
        - 'pred_instances'(InstanceData): Instances of model predictions (bboxes, masks, etc).
        - 'gt_tokens'(LabelData): Ground truth tokens for recognition.
        - 'pred_tokens'(LabelData): Predicted tokens for recognition.

    This structure supports both detection (bbox) and recognition (token) tasks,
    corresponding to the dual-head architecture of TableMaster model.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import InstanceData, LabelData
         >>> from structures.token_recog_data_sample import TableMasterDataSample
         >>> # Create data sample
         >>> data_sample = TableMasterDataSample()
         >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
         >>> 
         >>> # gt_instances for detection head (bboxes)
         >>> gt_instances = InstanceData(metainfo=img_meta)
         >>> gt_instances.bboxes = torch.rand((5, 4))
         >>> gt_instances.labels = torch.LongTensor([0, 1, 2, 3, 4])
         >>> data_sample.gt_instances = gt_instances
         >>> 
         >>> # gt_tokens for recognition head (tokens)
         >>> gt_tokens = LabelData()
         >>> gt_tokens.item = ['<SOS>', '<td>', 'text', '</td>', '<EOS>']
         >>> gt_tokens.padded_indexes = torch.LongTensor([0, 1, 2, 3, 4])
         >>> data_sample.gt_tokens = gt_tokens
         >>> 
         >>> len(data_sample.gt_instances)
         5
    """

    @property
    def gt_instances(self) -> InstanceData:
        """InstanceData: groundtruth instances for detection (bboxes, masks, etc)."""
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        """gt_instances setter."""
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        """gt_instances deleter."""
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        """InstanceData: prediction instances for detection (bboxes, masks, etc)."""
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        """pred_instances setter."""
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        """pred_instances deleter."""
        del self._pred_instances

    @property
    def gt_tokens(self) -> LabelData:
        """LabelData: ground truth tokens for recognition."""
        return self._gt_tokens

    @gt_tokens.setter
    def gt_tokens(self, value: LabelData) -> None:
        """gt_tokens setter."""
        self.set_field(value, '_gt_tokens', dtype=LabelData)

    @gt_tokens.deleter
    def gt_tokens(self) -> None:
        """gt_tokens deleter."""
        del self._gt_tokens

    @property
    def pred_tokens(self) -> LabelData:
        """LabelData: predicted tokens for recognition."""
        return self._pred_tokens

    @pred_tokens.setter
    def pred_tokens(self, value: LabelData) -> None:
        """pred_tokens setter."""
        self.set_field(value, '_pred_tokens', dtype=LabelData)

    @pred_tokens.deleter
    def pred_tokens(self) -> None:
        """pred_tokens deleter."""
        del self._pred_tokens