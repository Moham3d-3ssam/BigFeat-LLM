import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import setup.local_utils as local_utils
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from functools import partial


class BigFeat:
    """Base BigFeat Class for both classification and regression tasks"""

    def __init__(self, task_type='classification', n_trees=None,
                 operators=None, binary_operators=None, unary_operators=None):
        """
        Initialize the BigFeat object

        Parameters:
        -----------
        task_type : str, default='classification'
            The type of machine learning task. Either 'classification' or 'regression'.

        n_trees : int or None, default=None
            (N1) Number of trees used by the internal RandomForest models for
            feature-importance scoring. If None, falls back to the original
            behavior (sklearn's own default, n_estimators=100).

        operators : list or None, default=None
            (operator_portfolio) Full list of operator functions BigFeat may use.
            If None, falls back to the original hardcoded portfolio
            [multiply, add, subtract, abs, square].
            If you pass this, you should also pass binary_operators and/or
            unary_operators so BigFeat knows how to use each operator
            (how many operands it takes).

        binary_operators : list or None, default=None
            Subset of `operators` that take two operands (e.g. multiply, add,
            subtract). If None, falls back to the original defaults.

        unary_operators : list or None, default=None
            Subset of `operators` that take one operand (e.g. abs, square,
            original_feat passthrough). If None, falls back to the original
            defaults.

        Note: n_trees, operators, binary_operators, and unary_operators are
        all plain public attributes (self.n_trees, self.binary_operators,
        self.unary_operators). You can either pass them here, or set them
        directly on the instance after construction, e.g.:

            bf = BigFeat(task_type="classification")
            bf.n_trees = 200
            bf.binary_operators = [np.multiply, np.add]
            bf.unary_operators = [np.abs, np.square]
            bf.fit(X_train, y_train, iterations=10, ...)

        Both styles work identically: fit() always recomputes the combined
        self.operators list from whatever self.binary_operators and
        self.unary_operators hold at the moment fit() is called.
        """
        self.n_jobs = -1
        self.task_type = task_type

        # Validate task_type input
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be either 'classification' or 'regression'")

        # ---- N1: number of trees in the internal Random Forest models ----
        # If the user does not pass n_trees, keep the exact original behavior:
        # RandomForestClassifier/Regressor were previously called with no
        # n_estimators argument at all, i.e. sklearn's own default (100).
        self.n_trees = n_trees if n_trees is not None else 100

        # ---- operator_portfolio: which operators BigFeat may pick from ----
        # If the user customizes any of operators/binary_operators/unary_operators,
        # honor that. Otherwise, keep the exact original hardcoded portfolio.
        if operators is not None or binary_operators is not None or unary_operators is not None:
            self.binary_operators = binary_operators if binary_operators is not None \
                else [np.multiply, np.add, np.subtract]
            self.unary_operators = unary_operators if unary_operators is not None \
                else [np.abs, np.square, local_utils.original_feat]
            self.operators = operators if operators is not None \
                else (self.binary_operators + self.unary_operators)
        else:
            self.operators = [np.multiply, np.add, np.subtract, np.abs, np.square]
            self.binary_operators = [np.multiply, np.add, np.subtract]
            self.unary_operators = [np.abs, np.square, local_utils.original_feat]

    def fit(self, X, y, gen_size=5, random_state=0, iterations=5, estimator='avg',
            feat_imps=True, split_feats=None, check_corr=True, selection='stability',
            combine_res=True, N=None, alpha=None, eta=None, height=None,
            initial_importances=None):
        """
        Generate features using the training set.

        New optional hyperparameters (all default to the framework's original
        hardcoded behavior when not supplied):

        N : int or None, default=None
            Number of features to retain at each selection step. If None,
            falls back to the original behavior, i.e. N = number of input
            features (X.shape[1]).

        alpha : int or None, default=None
            Number of bootstrap-sampled tree models used for stability-based
            feature importance (previously the hardcoded `sample_count=1`
            inside get_feature_importances). If None, falls back to 1.

        eta : float or None, default=None
            Pearson correlation threshold above which a candidate feature is
            considered redundant and dropped (previously the hardcoded
            `cor_thresh = 0.8` inside check_correlations). If None, falls
            back to 0.8.

        height : int or None, default=None
            Maximum depth of the generated computation trees (previously the
            hardcoded `np.arange(3) + 1`, i.e. depths 1-3). If None, falls
            back to 3.

        initial_importances : array-like or None, default=None
            (I_f init) An externally supplied initial guess of each base
            feature's importance, one value per column of X, in the exact
            same column order as X. This is the vector the very first round
            of computation trees samples leaf features from (see Algorithm
            1, step 1 in the BigFeat paper). If None, falls back to the
            original behavior: importance is computed from a trained
            Random Forest / LightGBM model on X, y (when feat_imps=True),
            or uniform (when feat_imps=False). If provided, it OVERRIDES
            whatever was computed/defaulted above, after normalization
            (values are clipped to be non-negative and rescaled to sum to
            1). Providing this does not disable feat_imps or split_feats:
            those still run as before for split-based combination mining,
            but this vector wins as the final self.ig_vector used for
            feature sampling. This is only an initial seed - BigFeat still
            re-scores and updates importances on its own every iteration,
            exactly as it always has.
        """
        self.selection = selection
        # Recompute the combined operator list from whatever binary_operators
        # / unary_operators are CURRENTLY set on this instance. This matters
        # because these two lists can be set either through the constructor
        # OR as plain attribute assignment after construction
        # (e.g. bf.binary_operators = [...]) before calling fit(). Without
        # this recompute, self.operators would still reflect whatever was
        # set at __init__ time and silently ignore a later attribute change.
        self.operators = list(self.binary_operators) + list(self.unary_operators)
        self.imp_operators = np.ones(len(self.operators))
        self.operator_weights = self.imp_operators / self.imp_operators.sum()
        self.gen_steps = []
        self.n_feats = X.shape[1]
        self.n_rows = X.shape[0]

        # ---- N: number of features retained per selection step ----
        self.n_select = N if N is not None else self.n_feats
        # Safety clamp: N can never exceed the pool it is selected from
        # (n_feats * gen_size). Requesting more than that would otherwise
        # produce a smaller-than-expected selection later and break the
        # fixed-width per-iteration bookkeeping arrays with a shape
        # mismatch, so we cap it here instead of letting that surface as
        # a confusing ValueError deep in the iteration loop.
        max_pool = self.n_feats * gen_size
        if self.n_select > max_pool:
            self.n_select = max_pool

        # ---- alpha: number of stability-selection bootstrap models ----
        self.alpha = alpha if alpha is not None else 1

        # ---- eta: Pearson correlation redundancy threshold ----
        self.eta = eta if eta is not None else 0.8

        # ---- height: max depth of generated computation trees ----
        self.height = height if height is not None else 3

        self.ig_vector = np.ones(self.n_feats) / self.n_feats
        self.comb_mat = np.ones((self.n_feats, self.n_feats))
        self.split_vec = np.ones(self.n_feats)
        # Set RNG seed if provided for numpy
        self.rng = np.random.RandomState(seed=random_state)
        gen_feats = np.zeros((self.n_rows, self.n_feats * gen_size))
        # These arrays store the top self.n_select features kept from EACH
        # iteration, so their per-iteration width must be self.n_select, not
        # self.n_feats. N (self.n_select) is independent of the original
        # number of input columns, so this must not assume they're equal.
        iters_comb = np.zeros((self.n_rows, self.n_select * iterations))
        depths_comb = np.zeros(self.n_select * iterations)
        ids_comb = np.zeros(self.n_select * iterations, dtype=object)
        ops_comb = np.zeros(self.n_select * iterations, dtype=object)
        self.feat_depths = np.zeros(gen_feats.shape[1])
        self.depth_range = np.arange(self.height) + 1
        self.depth_weights = 1 / (2 ** self.depth_range)
        self.depth_weights /= self.depth_weights.sum()
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        if feat_imps:
            self.ig_vector, estimators = self.get_feature_importances(
                X, y, estimator, random_state, sample_count=self.alpha)
            # Guard against a degenerate importance vector (all zeros / NaN),
            # which would otherwise turn into 0/0 -> NaN and later crash
            # np.random.choice's p= argument.
            if not np.isfinite(self.ig_vector).all() or self.ig_vector.sum() <= 0:
                self.ig_vector = np.ones(self.n_feats) / self.n_feats
            else:
                self.ig_vector /= self.ig_vector.sum()
            for tree in estimators:
                paths = self.get_paths(tree, np.arange(X.shape[1]))
                self.get_split_feats(paths, self.split_vec)
            self.split_vec /= self.split_vec.sum()
            # self.split_vec = StandardScaler().fit_transform(self.split_vec.reshape(1, -1), {'var_':5})
            if split_feats == "comb":
                self.ig_vector = np.multiply(self.ig_vector, self.split_vec)
                self.ig_vector /= self.ig_vector.sum()
            elif split_feats == "splits":
                self.ig_vector = self.split_vec

        # ---- initial_importances: externally supplied I_f override ----
        # This runs regardless of feat_imps, so it works whether or not the
        # internal Random Forest/LightGBM importance computation above ran.
        # It always wins as the final seed used for feature sampling.
        if initial_importances is not None:
            init_imp = np.asarray(initial_importances, dtype=float)
            if init_imp.shape[0] != self.n_feats:
                raise ValueError(
                    f"initial_importances has {init_imp.shape[0]} values but "
                    f"X has {self.n_feats} columns; they must match exactly "
                    f"and be in the same column order."
                )
            # Negative importances are not meaningful as sampling
            # probabilities; clip them to zero rather than erroring, since
            # an LLM or user may pass a raw signed score.
            init_imp = np.clip(init_imp, a_min=0, a_max=None)
            if not np.isfinite(init_imp).all() or init_imp.sum() <= 0:
                # Degenerate override (all zero / NaN): ignore it and keep
                # whatever self.ig_vector already holds (computed above, or
                # the uniform default set earlier), rather than crashing.
                pass
            else:
                self.ig_vector = init_imp / init_imp.sum()

        for iteration in range(iterations):
            self.tracking_ops = []
            self.tracking_ids = []
            gen_feats = np.zeros((self.n_rows, self.n_feats * gen_size))
            self.feat_depths = np.zeros(gen_feats.shape[1])
            for i in range(gen_feats.shape[1]):
                dpth = self.rng.choice(self.depth_range, p=self.depth_weights)
                ops = []
                ids = []
                gen_feats[:, i] = self.feat_with_depth(X, dpth, ops, ids)  # ops and ids are updated
                self.feat_depths[i] = dpth
                self.tracking_ops.append(ops)
                self.tracking_ids.append(ids)
            self.tracking_ids = np.array(self.tracking_ids + [[]], dtype='object')[:-1]
            self.tracking_ops = np.array(self.tracking_ops + [[]], dtype='object')[:-1]
            imps, estimators = self.get_feature_importances(
                gen_feats, y, estimator, random_state, sample_count=self.alpha)
            total_feats = np.argsort(imps)
            feat_args = total_feats[-self.n_select:]
            gen_feats = gen_feats[:, feat_args]
            self.tracking_ids = self.tracking_ids[feat_args]
            self.tracking_ops = self.tracking_ops[feat_args]
            self.feat_depths = self.feat_depths[feat_args]
            depths_comb[iteration * self.n_select:(iteration + 1) * self.n_select] = self.feat_depths
            ids_comb[iteration * self.n_select:(iteration + 1) * self.n_select] = self.tracking_ids
            ops_comb[iteration * self.n_select:(iteration + 1) * self.n_select] = self.tracking_ops
            iters_comb[:, iteration * self.n_select:(iteration + 1) * self.n_select] = gen_feats
            for i, op in enumerate(self.operators):
                for feat in self.tracking_ops:
                    for feat_op in feat:
                        if op == feat_op[0]:
                            self.imp_operators[i] += 1
            self.operator_weights = self.imp_operators / self.imp_operators.sum()
        if selection == 'stability' and iterations > 1 and combine_res:
            imps, estimators = self.get_feature_importances(
                iters_comb, y, estimator, random_state, sample_count=self.alpha)
            total_feats = np.argsort(imps)
            feat_args = total_feats[-self.n_select:]
            gen_feats = iters_comb[:, feat_args]
            self.tracking_ids = ids_comb[feat_args]
            self.tracking_ops = ops_comb[feat_args]
            self.feat_depths = depths_comb[feat_args]

        if selection == 'stability' and check_corr:
            gen_feats, to_drop_cor = self.check_correlations(gen_feats)
            self.tracking_ids = np.delete(self.tracking_ids, to_drop_cor)
            self.tracking_ops = np.delete(self.tracking_ops, to_drop_cor)
            self.feat_depths = np.delete(self.feat_depths, to_drop_cor)
        gen_feats = np.hstack((gen_feats, X))

        if selection == 'fAnova':
            # Use the appropriate feature selection method based on task type
            if self.task_type == 'classification':
                self.fAnova_best = SelectKBest(f_classif, k=self.n_select)
            else:  # regression
                self.fAnova_best = SelectKBest(f_regression, k=self.n_select)
            gen_feats = self.fAnova_best.fit_transform(gen_feats, y)

        return gen_feats

    def transform(self, X):
        """ Produce features from the fitted BigFeat object """
        X = self.scaler.transform(X)
        self.n_rows = X.shape[0]
        gen_feats = np.zeros((self.n_rows, len(self.tracking_ids)))
        for i in range(gen_feats.shape[1]):
            dpth = self.feat_depths[i]
            op_ls = self.tracking_ops[i].copy()
            id_ls = self.tracking_ids[i].copy()
            gen_feats[:, i] = self.feat_with_depth_gen(X, dpth, op_ls, id_ls)
        gen_feats = np.hstack((gen_feats, X))
        if self.selection == 'fAnova':
            gen_feats = self.fAnova_best.transform(gen_feats)
        return gen_feats

    def select_estimator(self, X, y, estimators_names=None):
        """
        Select the best estimator based on cross-validation

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        estimators_names : list or None
            List of estimator names to try. If None, uses appropriate defaults.

        Returns:
        --------
        model : estimator
            Fitted best estimator
        """
        # Use appropriate default estimators based on task type
        if estimators_names is None:
            if self.task_type == 'classification':
                estimators_names = ['dt', 'lr']
            else:  # regression
                estimators_names = ['dt_reg', 'lr_reg']

        # Define available estimators based on task type
        estimators_dic = {
            # Classification estimators
            'dt': DecisionTreeClassifier(),
            'lr': LogisticRegression(),
            'rf': RandomForestClassifier(n_estimators=self.n_trees, n_jobs=self.n_jobs),
            'lgb': LGBMClassifier(),
            # Regression estimators
            'dt_reg': DecisionTreeRegressor(),
            'lr_reg': LinearRegression(),
            'rf_reg': RandomForestRegressor(n_estimators=self.n_trees, n_jobs=self.n_jobs),
            'lgb_reg': LGBMRegressor()
        }

        models_score = {}

        for estimator in estimators_names:
            model = estimators_dic[estimator]

            # Use appropriate scoring metric based on task type
            if self.task_type == 'classification':
                scorer = make_scorer(f1_score)
            else:  # regression
                scorer = make_scorer(r2_score)

            models_score[estimator] = cross_val_score(model, X, y, cv=3, scoring=scorer).mean()

        best_estimator = max(models_score, key=models_score.get)
        best_model = estimators_dic[best_estimator]
        best_model.fit(X, y)
        return best_model

    def get_feature_importances(self, X, y, estimator, random_state, sample_count=1, sample_size=3, n_jobs=1):
        """Return feature importances by specified method"""

        importance_sum = np.zeros(X.shape[1])
        total_estimators = []
        for sampled in range(sample_count):
            sampled_ind = np.random.choice(np.arange(self.n_rows), size=self.n_rows // sample_size, replace=False)
            sampled_X = X[sampled_ind]
            sampled_y = np.take(y, sampled_ind)

            # Different behavior based on task type
            if estimator == "rf":
                if self.task_type == 'classification':
                    estm = RandomForestClassifier(n_estimators=self.n_trees, random_state=random_state, n_jobs=n_jobs)
                else:  # regression
                    estm = RandomForestRegressor(n_estimators=self.n_trees, random_state=random_state, n_jobs=n_jobs)

                estm.fit(sampled_X, sampled_y)
                total_importances = estm.feature_importances_
                estimators = estm.estimators_
                total_estimators += estimators

            elif estimator == "avg":
                # For classification
                if self.task_type == 'classification':
                    clf = RandomForestClassifier(n_estimators=self.n_trees, random_state=random_state, n_jobs=n_jobs)
                    clf.fit(sampled_X, sampled_y)
                    rf_importances = clf.feature_importances_
                    estimators = clf.estimators_
                    total_estimators += estimators

                    # LightGBM for classification
                    train_data = lgb.Dataset(sampled_X, label=sampled_y)
                    param = {'num_leaves': 31, 'objective': 'binary', 'verbose': -1}
                    param['metric'] = 'auc'

                # For regression
                else:
                    clf = RandomForestRegressor(n_estimators=self.n_trees, random_state=random_state, n_jobs=n_jobs)
                    clf.fit(sampled_X, sampled_y)
                    rf_importances = clf.feature_importances_
                    estimators = clf.estimators_
                    total_estimators += estimators

                    # LightGBM for regression
                    train_data = lgb.Dataset(sampled_X, label=sampled_y)
                    param = {'num_leaves': 31, 'objective': 'regression', 'verbose': -1}
                    param['metric'] = 'rmse'

                # Common LightGBM code for both tasks
                num_round = 2
                bst = lgb.train(param, train_data, num_round)
                lgb_imps = bst.feature_importance(importance_type='gain')
                lgb_sum = lgb_imps.sum()
                # LightGBM can return all-zero gain (e.g. if it found no
                # useful split), which would otherwise produce 0/0 -> NaN.
                if lgb_sum > 0 and np.isfinite(lgb_sum):
                    lgb_imps = lgb_imps / lgb_sum
                else:
                    lgb_imps = np.ones(X.shape[1]) / X.shape[1]
                total_importances = (rf_importances + lgb_imps) / 2

            # Final safety net: never let a NaN or all-zero importance
            # vector leave this function.
            if not np.isfinite(total_importances).all() or total_importances.sum() == 0:
                total_importances = np.ones(X.shape[1]) / X.shape[1]

            importance_sum += total_importances
        return importance_sum, total_estimators

    def get_weighted_feature_importances(self, X, y, estimator, random_state):
        """Return feature importances weighted by model performance"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state)

        # Choose appropriate model based on task type
        if self.task_type == 'classification':
            estm = RandomForestClassifier(n_estimators=self.n_trees, random_state=random_state, n_jobs=self.n_jobs)
        else:  # regression
            estm = RandomForestRegressor(n_estimators=self.n_trees, random_state=random_state, n_jobs=self.n_jobs)

        estm.fit(X_train, y_train)
        ests = estm.estimators_
        model = estm
        imps = np.zeros((len(model.estimators_), X.shape[1]))
        scores = np.zeros(len(model.estimators_))

        for i, each in enumerate(model.estimators_):
            # Different scoring metrics based on task type
            if self.task_type == 'classification':
                y_probas_train = each.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_probas_train)
            else:  # regression
                y_pred_train = each.predict(X_test)
                score = r2_score(y_test, y_pred_train)

            imps[i] = each.feature_importances_
            scores[i] = score

        weights = scores / scores.sum()
        return np.average(imps, axis=0, weights=weights)

    def feat_with_depth(self, X, depth, op_ls, feat_ls):
        """ Recursively generate a new features """
        if depth == 0:
            feat_ind = self.rng.choice(np.arange(len(self.ig_vector)), p=self.ig_vector)
            feat_ls.append(feat_ind)
            return X[:, feat_ind]
        depth -= 1
        op = self.rng.choice(self.operators, p=self.operator_weights)
        if op in self.binary_operators:
            feat_1 = self.feat_with_depth(X, depth, op_ls, feat_ls)
            feat_2 = self.feat_with_depth(X, depth, op_ls, feat_ls)
            op_ls.append((op, depth))
            return op(feat_1, feat_2)
        elif op in self.unary_operators:
            feat_1 = self.feat_with_depth(X, depth, op_ls, feat_ls)
            op_ls.append((op, depth))
            return op(feat_1)

    def feat_with_depth_gen(self, X, depth, op_ls, feat_ls):
        """ Reproduce generated features with new data """
        if depth == 0:
            feat_ind = feat_ls.pop()
            return X[:, feat_ind]
        depth -= 1
        op = op_ls.pop()[0]
        if op in self.binary_operators:
            feat_1 = self.feat_with_depth_gen(X, depth, op_ls, feat_ls)
            feat_2 = self.feat_with_depth_gen(X, depth, op_ls, feat_ls)
            return op(feat_2, feat_1)
        elif op in self.unary_operators:
            feat_1 = self.feat_with_depth_gen(X, depth, op_ls, feat_ls)
            return op(feat_1)

    def check_correlations(self, feats):
        """ Check correlations among the selected features """
        cor_thresh = self.eta
        corr_matrix = pd.DataFrame(feats).corr().abs()
        mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
        tri_df = corr_matrix.mask(mask)
        to_drop = [c for c in tri_df.columns if any(tri_df[c] > cor_thresh)]
        # remove the feature with lower importance if corr > cor_thresh
        # to_drop = []
        # for c in tri_df.columns:
        #     if any(corr_matrix[c] > cor_thresh):
        #         for c_, cor_val in enumerate(corr_matrix[c].values):
        #             if cor_val > cor_thresh and c != c_:
        #                 if self.ig_vector_gen[c_] < self.ig_vector_gen[c] and c_ not in to_drop:
        #                     to_drop.append(c_)

        feats = pd.DataFrame(feats).drop(to_drop, axis=1)
        return feats.values, to_drop

    def get_paths(self, clf, feature_names):
        """ Returns every path in the decision tree"""
        tree_ = clf.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        path = []
        path_list = []

        def recurse(node, depth, path_list):
            if tree_.feature[node] == _tree.TREE_UNDEFINED:
                path_list.append(path.copy())
            else:
                name = feature_name[node]
                path.append(name)
                recurse(tree_.children_left[node], depth + 1, path_list)
                recurse(tree_.children_right[node], depth + 1, path_list)
                path.pop()

        recurse(0, 1, path_list)

        new_list = []
        for i in range(len(path_list)):
            if path_list[i] != path_list[i - 1]:
                new_list.append(path_list[i])
        return new_list

    def get_combos(self, paths, comb_mat):
        """ Fills Combination matrix with values """
        for i in range(len(comb_mat)):
            for pt in paths:
                if i in pt:
                    comb_mat[i][pt] += 1

    def get_split_feats(self, paths, split_vec):
        """ Fills split vector with values """
        for i in range(len(split_vec)):
            for pt in paths:
                if i in pt:
                    split_vec[i] += 1