# Plan: Add Random X,Y Translation Augmentation

## Summary
Add a random translation augmentation to `src/data/augmentations.py` that shifts images horizontally and vertically by a percentage of image dimensions.

## Implementation Details

### Changes to `src/data/augmentations.py`

Add translation configuration parameters:
- `translate_limit`: float, max translation as percentage (e.g., 0.2 = ±20%)
- `translate_prob`: float, probability of applying translation

Insert after the random scale transform (line 18-19), using `A.Affine`:

```python
A.Affine(
    translate_percent={"x": (-translate_limit, translate_limit), "y": (-translate_limit, translate_limit)},
    cval=0,  # black fill
    p=translate_prob,
)
```

The `A.Affine` transform automatically handles:
- Both images and bounding boxes (when `bbox_aware=True`)
- Black border fill via `cval=0`

### Config Updates

Add defaults to config files:
- `translate_limit: 0.2`
- `translate_prob: 0.5`

## Verification
- Transform applies correctly in training pipeline
- Bounding boxes shift with image when `bbox_aware=True`
- Empty regions filled with black (0)
