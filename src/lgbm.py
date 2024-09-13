import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight

class LGBM:
    def __init__(self, config, logger):
        self._learn_params = config["LEARN_PARAMS"]
        self._model_params = config["MODEL_PARAMS"]
        self._weight = None
        lgb.register_logger(logger)

    def train(self, X_train, y_train, X_valid, y_valid, categorical_feature=None):
        if 'class_weight' in self._model_params:
            self._logger.info(f"Computing sample weight with class_weight: {self._model_params['class_weight']}")
            self._weight = compute_sample_weight(self._model_params['class_weight'], y_train)
            # remove class_weight from model_params
            self._model_params.pop('class_weight')

        train_X = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature, weight=self._weight)
        valid_X = lgb.Dataset(X_valid, y_valid, categorical_feature=categorical_feature)

        self._model = lgb.train(
            self._model_params,
            train_X,
            num_boost_round=self._learn_params["num_boost_round"],
            valid_sets=[train_X, valid_X],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self._learn_params["early_stopping_rounds"], verbose=True),
                lgb.log_evaluation(period=self._learn_params["log_evaluation_period"]),
            ]
        )

    def predict(self, X):
        return self._model.predict(X)

    def save_model(self, path):
        self._model.save_model(path)

    def load_model(self, path):
        self._model = lgb.Booster(model_file=path)
