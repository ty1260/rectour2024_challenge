import polars as pl
import gc
from sklearn.preprocessing import LabelEncoder

from lgbm import LGBM
from evaluator import calc_mrr
from logger import set_logger


exp_path = './result/'

# set config
config = {
    'NEGATIVE_SAMPLING': 'Random',
    'NEGATIVE_RATIO': 20,
    'LEARN_PARAMS': {
        'num_boost_round': 10000,
        'early_stopping_rounds': 100,
        'log_evaluation_period': 100,
    },
    'MODEL_PARAMS': {
        'objective': 'binary',
        'metric': 'auc',
        'n_jobs': -1,
        'random_state': 71
    }
}

# make logger
logger = set_logger(exp_path, 'train.log')


# load dataset
data_dir = "./accommodation-reviews/"

train_users = pl.read_csv(data_dir + "rectour24/train_users.csv")
train_reviews = pl.read_csv(data_dir + "preprocessed/train_reviews.csv")
train_matches = pl.read_csv(data_dir + "rectour24/train_matches.csv")

valid_users = pl.read_csv(data_dir + "rectour24/val_users.csv")
valid_reviews = pl.read_csv(data_dir + "preprocessed/val_reviews.csv")
valid_matches = pl.read_csv(data_dir + "rectour24/val_matches.csv")

test_users = pl.read_csv(data_dir + "rectour24/test_users.csv")
test_reviews = pl.read_csv(data_dir + "preprocessed/test_reviews.csv")

users_tfidf = pl.read_csv(data_dir + "preprocessed/user_tfidf_ica.csv")
reviews_tfidf = pl.read_csv(data_dir + "preprocessed/review_tfidf_ica.csv")


# make all candidates
train_all = train_reviews.join(train_users, on='accommodation_id', how='inner')
valid_all = valid_reviews.join(valid_users, on='accommodation_id', how='inner')
test_all = test_reviews.join(test_users, on='accommodation_id', how='inner')

# add ground truth
train_matches = train_matches.with_columns(gt=pl.lit(1))
valid_matches = valid_matches.with_columns(gt=pl.lit(1))

train_all = train_all.join(train_matches, on=['accommodation_id', 'user_id', 'review_id'], how='left')
valid_all = valid_all.join(valid_matches, on=['accommodation_id', 'user_id', 'review_id'], how='left')

# gt fill na
train_all = train_all.with_columns(gt=pl.col('gt').fill_null(0))
valid_all = valid_all.with_columns(gt=pl.col('gt').fill_null(0))


# label encoding
category_features = ['guest_type', 'guest_country', 'accommodation_type', 'accommodation_country']

for col in category_features:
    le = LabelEncoder()
    all_data = pl.concat([train_all[col], valid_all[col], test_all[col]])
    le.fit(all_data.to_numpy())

    train_all = train_all.with_columns(pl.Series(le.transform(train_all[col].to_numpy())).alias(col))
    valid_all = valid_all.with_columns(pl.Series(le.transform(valid_all[col].to_numpy())).alias(col))
    test_all = test_all.with_columns(pl.Series(le.transform(test_all[col].to_numpy())).alias(col))


# downsampling
train_all = pl.concat([
    train_all.filter(pl.col('gt') == 1),
    train_all.filter(pl.col('gt') == 0).sample(n=len(train_all.filter(pl.col('gt') == 1))*config['NEGATIVE_RATIO'], seed=71)
])

logger.info(f"train gt distribution: 'gt' = 1: {len(train_all.filter(pl.col('gt') == 1))}, 'gt' = 0: {len(train_all.filter(pl.col('gt') == 0))}")

del train_matches, valid_matches, all_data, train_users, train_reviews, valid_users, valid_reviews, test_users, test_reviews
gc.collect()


# add tfidf features
train_all = train_all.join(users_tfidf, on=['user_id', 'accommodation_id'], how='inner')
train_all = train_all.join(reviews_tfidf, on=['review_id', 'accommodation_id'], how='inner')

valid_all = valid_all.join(users_tfidf, on=['user_id', 'accommodation_id'], how='inner')
valid_all = valid_all.join(reviews_tfidf, on=['review_id', 'accommodation_id'], how='inner')

test_all = test_all.join(users_tfidf, on=['user_id', 'accommodation_id'], how='inner')
test_all = test_all.join(reviews_tfidf, on=['review_id', 'accommodation_id'], how='inner')


# drop columns
drop_cols = [
    'user_id',
    'review_id',
    'accommodation_id',
    'review_title',
    'review_positive',
    'review_negative',
]

train_X = train_all.drop(drop_cols + ['gt']).to_pandas()
valid_X = valid_all.drop(drop_cols + ['gt']).to_pandas()
test_X = test_all.drop(drop_cols).to_pandas()


# train
train_y = train_all['gt'].to_pandas()
valid_y = valid_all['gt'].to_pandas()

lgbm = LGBM(config, logger)
lgbm.train(train_X, train_y, valid_X, valid_y, categorical_feature=category_features)


# predict
valid_pred = lgbm.predict(valid_X)
test_pred = lgbm.predict(test_X)

valid_preds = valid_all.select(['accommodation_id', 'user_id', 'review_id', 'gt'])
valid_preds = valid_preds.with_columns(pl.Series(valid_pred).alias('score'))

test_preds = test_all.select(['accommodation_id', 'user_id', 'review_id'])
test_preds = test_preds.with_columns(pl.Series(test_pred).alias('score'))


# select top 10
valid_preds = valid_preds.with_columns(rank=pl.col('score').rank(method='ordinal', descending=True).over('accommodation_id', 'user_id'))
valid_preds = valid_preds.filter(pl.col('rank') <= 10).sort('rank')

test_preds = test_preds.with_columns(rank=pl.col('score').rank(method='ordinal', descending=True).over('accommodation_id', 'user_id'))
test_preds = test_preds.filter(pl.col('rank') <= 10).sort('rank')

mrr = calc_mrr(valid_preds)
logger.info(f'MRR: {mrr}')


# make submission file
submissions = test_preds.group_by(['accommodation_id', 'user_id']).agg(pl.col('review_id'))

for i in range(10):
    submissions = submissions.with_columns(pl.col('review_id').list.get(i).alias(f'review_{i+1}'))

submissions.drop('review_id').to_pandas().to_csv(f'{exp_path}/test_submission.csv.zip', compression='zip')
