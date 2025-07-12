# Copyright (c) Lê Hoàn Chân. All rights reserved.
from mmengine.structures import BaseDataElement, LabelData

class TokenRecogDataSample(BaseDataElement):
    """A data structure interface of MMOCR for token recognition.

    Attributes:
        gt_tokens (LabelData): Ground truth tokens.
        pred_tokens (LabelData): Predicted tokens.

    Examples:
        >>> import numpy as np
        >>> from mmengine.structures import LabelData
        >>> from mmocr.data import TokenRecogDataSample
        >>> data_sample = TokenRecogDataSample()
        >>> data_sample.gt_tokens = LabelData(item=['token1', 'token2'])
        >>> data_sample.pred_tokens = LabelData(item=['token1', 'tokenX'])
        >>> print(data_sample)
        <TokenRecogDataSample(
            META INFORMATION
            DATA FIELDS
            gt_tokens: <LabelData(
                    META INFORMATION
                    DATA FIELDS
                    item: ['token1', 'token2']
                ) at ...>
            pred_tokens: <LabelData(
                    META INFORMATION
                    DATA FIELDS
                    item: ['token1', 'tokenX']
                ) at ...>
        ) at ...>
    """

    @property
    def gt_tokens(self) -> LabelData:
        """LabelData: ground truth tokens."""
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
        """LabelData: predicted tokens."""
        return self._pred_tokens

    @pred_tokens.setter
    def pred_tokens(self, value: LabelData) -> None:
        """pred_tokens setter."""
        self.set_field(value, '_pred_tokens', dtype=LabelData)

    @pred_tokens.deleter
    def pred_tokens(self) -> None:
        """pred_tokens deleter."""
        del self._pred_tokens