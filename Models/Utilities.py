import numpy as np

class Utilities:
    @staticmethod
    def tune_random_forest(X_train, y_train, X_val, y_val, configs):
        from Ensemble import RandomForest

        results = []

        for (n_estimators, max_depth, min_samples_split) in configs:
            rf = RandomForest(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split
            )

            rf.fit(X_train, y_train)

            train_acc = rf.accuracy(X_train, y_train)
            val_acc = rf.accuracy(X_val, y_val)

            results.append((n_estimators, max_depth, min_samples_split, train_acc, val_acc))

            print(f"n={n_estimators}, depth={max_depth}, min_split={min_samples_split} "
                f"| Train={train_acc:.4f} | Val={val_acc:.4f}")

        return results

    @staticmethod
    def tune_knn(X_train, y_train, X_val, y_val, k_values):
        results = []

        for k in k_values:
            from KNN import KNN
            model = KNN(k=k)
            model.fit(X_train, y_train)

            val_acc = model.accuracy(X_val, y_val)
            results.append((k, val_acc))

            print(f"k={k}, Val Acc={val_acc:.4f}")

        return results


    @staticmethod
    def tune_logistic_lambda(X_train, y_train, X_val, y_val, lambda_values, lr=0.01, epochs=200):
        results = []

        from logistic_regression_ovr import LogisticRegressionOVR

        for lam in lambda_values:
            model = LogisticRegressionOVR(
                learning_rate=lr,
                epochs=epochs,
                lambda_=lam
            )

            model.fit(X_train, y_train)
            val_acc = model.accuracy(X_val, y_val)

            results.append((lam, val_acc))

            print(f"lambda={lam}, Val Acc={val_acc:.4f}")

        return results


    @staticmethod
    def learning_curve(model_class, X_train, y_train, X_val, y_val, sizes, **model_kwargs):
        train_scores = []
        val_scores = []

        for s in sizes:
            model = model_class(**model_kwargs)

            X_sub = X_train[:s]
            y_sub = y_train[:s]

            model.fit(X_sub, y_sub)

            train_acc = model.accuracy(X_sub, y_sub)
            val_acc = model.accuracy(X_val, y_val)

            train_scores.append(train_acc)
            val_scores.append(val_acc)

            print(f"Size={s} | Train={train_acc:.4f} | Val={val_acc:.4f}")

        return train_scores, val_scores
    

    
         
    