# Copyright (c) Lê Hoàn Chân. All rights reserved.
from mmengine.structures import BaseDataElement, LabelData

class TokenRecogDataSample(BaseDataElement):
    """A data structure interface of MMOCR for token recognition. They are used
    as interfaces between different components.

    The attributes in ``TokenRecogDataSample`` are divided into two parts:

        - ``gt_token``(LabelData): Ground truth token.
        - ``pred_token``(LabelData): predictions token.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import LabelData
         >>> from mmocr.data import TokenRecogDataSample
         >>> # gt_token
         >>> data_sample = TokenRecogDataSample()
         >>> img_meta = dict(img_shape=(800, 1196, 3),
         ...                 pad_shape=(800, 1216, 3))
         >>> gt_token = LabelData(metainfo=img_meta)
         >>> gt_token.item = 'mmocr'
         >>> data_sample.gt_token = gt_token
         >>> assert 'img_shape' in data_sample.gt_token.metainfo_keys()
         >>> print(data_sample)
         <TokenRecogDataSample(
             META INFORMATION
             DATA FIELDS
             gt_token: <LabelData(
                     META INFORMATION
                     pad_shape: (800, 1216, 3)
                     img_shape: (800, 1196, 3)
                     DATA FIELDS
                     item: 'mmocr'
                 ) at 0x7f21fb1b9190>
         ) at 0x7f21fb1b9880>
         >>> # pred_token
         >>> pred_token = LabelData(metainfo=img_meta)
         >>> pred_token.item = 'mmocr'
         >>> data_sample = TokenRecogDataSample(pred_token=pred_token)
         >>> assert 'pred_token' in data_sample
         >>> data_sample = TokenRecogDataSample()
         >>> gt_token_data = dict(item='mmocr')
         >>> gt_token = LabelData(**gt_token_data)
         >>> data_sample.gt_token = gt_token
         >>> assert 'gt_token' in data_sample
         >>> assert 'item' in data_sample.gt_token
    """

    @property
    def gt_token(self) -> LabelData:
        """LabelData: ground truth token.
        """
        return self._gt_token

    @gt_token.setter
    def gt_token(self, value: LabelData) -> None:
        """gt_token setter."""
        self.set_field(value, '_gt_token', dtype=LabelData)

    @gt_token.deleter
    def gt_token(self) -> None:
        """gt_token deleter."""
        del self._gt_token

    @property
    def pred_token(self) -> LabelData:
        """LabelData: prediction token.
        """
        return self._pred_token

    @pred_token.setter
    def pred_token(self, value: LabelData) -> None:
        """pred_token setter."""
        self.set_field(value, '_pred_token', dtype=LabelData)

    @pred_token.deleter
    def pred_token(self) -> None:
        """pred_token deleter."""
        del self._pred_token