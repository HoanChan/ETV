# Copyright (c) Lê Hoàn Chân. All rights reserved.
from mmengine.structures import BaseDataElement, LabelData

class TokenRecogDataSample(BaseDataElement):
    """A data structure interface of MMOCR for token recognition.

    Attributes:
        gt_tokens (LabelData): Ground truth tokens.
        pred_tokens (LabelData): Predicted tokens.
        gt_bboxs (LabelData): Ground truth bounding boxes.
        pred_bboxs (LabelData): Predicted bounding boxes.

    Examples:
        >>> import numpy as np
        >>> from mmengine.structures import LabelData
        >>> from mmocr.data import TokenRecogDataSample
        >>> data_sample = TokenRecogDataSample()
        >>> data_sample.gt_tokens = LabelData(item=['token1', 'token2'])
        >>> data_sample.pred_tokens = LabelData(item=['token1', 'tokenX'])
        >>> data_sample.gt_bboxs = LabelData(item=np.array([[0, 0, 10, 10]]))
        >>> data_sample.pred_bboxs = LabelData(item=np.array([[1, 1, 11, 11]]))
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
            gt_bboxs: <LabelData(
                    META INFORMATION
                    DATA FIELDS
                    item: [[ 0  0 10 10]]
                ) at ...>
            pred_bboxs: <LabelData(
                    META INFORMATION
                    DATA FIELDS
                    item: [[ 1  1 11 11]]
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

    @property
    def gt_bboxs(self) -> LabelData:
        """LabelData: ground truth bounding boxes."""
        return self._gt_bboxs

    @gt_bboxs.setter
    def gt_bboxs(self, value: LabelData) -> None:
        """gt_bboxs setter."""
        self.set_field(value, '_gt_bboxs', dtype=LabelData)

    @gt_bboxs.deleter
    def gt_bboxs(self) -> None:
        """gt_bboxs deleter."""
        del self._gt_bboxs

    @property
    def pred_bboxs(self) -> LabelData:
        """LabelData: predicted bounding boxes."""
        return self._pred_bboxs

    @pred_bboxs.setter
    def pred_bboxs(self, value: LabelData) -> None:
        """pred_bboxs setter."""
        self.set_field(value, '_pred_bboxs', dtype=LabelData)

    @pred_bboxs.deleter
    def pred_bboxs(self) -> None:
        """pred_bboxs deleter."""
        del self._pred_bboxs