import polars as pl

def get_top_k(preds: pl.DataFrame, k: int) -> pl.DataFrame:
    pass

def calc_mrr(preds: pl.DataFrame) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for the given predictions.

    Args:
        preds: Predictions DataFrame with columns ['accommodation_id', 'user_id', 'review_id', 'score', 'gt']

    Returns:
        MRR score
    """
    preds = preds.sort(['accommodation_id', 'user_id', 'score', 'rank'], descending=[False, False, True, False])

    # calculate reciprocal rank
    preds = preds.with_columns(rr=pl.col('gt')/pl.col('rank'))
    mrr = (
        preds
        .group_by(['accommodation_id', 'user_id'])
        .agg(pl.col('rr').mean().alias('mrr'))
        .select('mrr')
        .to_series()[0]
    )

    return mrr
