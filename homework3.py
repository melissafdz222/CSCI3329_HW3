
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_classif
from itertools import combinations
import warnings
import time
warnings.filterwarnings('ignore')

#Set random seed
RANDOM_STATE = 17342
np.random.seed(RANDOM_STATE)

#PART 1
def load_and_preprocess():
    #Load the three csv data files
    train_df = pd.read_csv('train.csv', header=None)
    val_df = pd.read_csv('validation.csv', header=None)
    test_df = pd.read_csv('test.csv', header=None)
    
    print("\n" + "o" * 70)
    print("PART 1: Dataset Loading and Preprocessing")
    print("o" * 70)
    
    print(f"\nDataset shapes:\n")
    print(f"  Training set:   {train_df.shape}")
    print(f"  Validation set: {val_df.shape}")
    print(f"  Test set:       {test_df.shape}")
    
    #Combined all data for analysis
    train_df['source'] = 'train'
    val_df['source'] = 'validation'
    test_df['source'] = 'test'
    
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    feature_cols = [f'F{i+1}' for i in range(51)]
    target_cols = ['H1', 'H2', 'H3', 'H4', 'H5', 'H0']
    
    #Assign column names
    all_data.columns = feature_cols + target_cols + ['source']
    
    #Extract features and targets
    X = all_data[feature_cols].copy()
    y_raw = all_data[target_cols].copy()
    sources = all_data['source']
    
    #Create multi class target: 0=H0(decline), 1-5=H1-H5
    y = pd.Series(index=y_raw.index, dtype=int)
    
    for i, row in y_raw.iterrows():
        best_found = None
        for j, col in enumerate(['H1', 'H2', 'H3', 'H4', 'H5']):
            if row[col] == 1:
                best_found = j + 1  #1-5
                break
        if best_found is not None:
            y[i] = best_found
        elif row['H0'] == 1:
            y[i] = 0
        else:
            #If no +1 found, uses H0 as default
            y[i] = 0
    
    #Check class distribution
    print("\nClass Distribution (after combining all sets):")
    class_counts = y.value_counts().sort_index()
    class_names = ['H0 (Decline)', 'H1', 'H2', 'H3', 'H4', 'H5']
    for idx, count in class_counts.items():
        i = int(idx)
        print(f"  {class_names[i]}: {count} ({count/len(y)*100:.1f}%)")
    
    #Check for missing values
    print(f"\nMissing values in features: {X.isnull().sum().sum()}")
    
    #Remove any rows with missing values
    X = X.dropna()
    y = y[X.index]
    sources = sources[X.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    print(f"\nFinal dataset shape: {X_scaled.shape}")
    print(f"\nNumber of classes: {len(np.unique(y))}")
    
    #Split back into train/validation/test
    train_mask = sources == 'train'
    val_mask = sources == 'validation'
    test_mask = sources == 'test'
    
    X_train = X_scaled[train_mask]
    X_val = X_scaled[val_mask]
    X_test = X_scaled[test_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
    
    print(f"\nFinal splits:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler

#PART 2
def evaluate_classifiers(X_train, y_train, X_test, y_test):
    
    print("\n" + "o" * 70)
    print("PART 2: Algorithm Comparison")
    print("o" * 70)
    
    #Define the five classifiers
    models = {
        'Perceptron': Perceptron(max_iter=1000, random_state=RANDOM_STATE),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Gaussian Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, verbose=False)
    }
    
    #Setup cross validation 
    print("\nRunning 10x100 cross validation (1,000 evaluations per algorithm)")
    print("Note: This might take a few seconds\n")
    
    rkf = RepeatedKFold(n_splits=10, n_repeats=100, random_state=RANDOM_STATE)
    results = {}

    print("-" * 78)
    print(f"{'Algorithm':<25} {'Mean Accuracy':<15} {'Std Dev':<10} {'Test Accuracy':<15} {'Time (s)':<10}")
    print("-" * 78)
    
    for name, model in models.items():
        print(f"  Running {name}...", end=' ', flush=True)
        start_time = time.time()
        
        #Cross validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=rkf, 
                                    scoring='accuracy', n_jobs=-1)
        mean_cv = cv_scores.mean()
        std_cv = cv_scores.std()
        
        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        elapsed_time = time.time() - start_time
        
        results[name] = {
            'cv_mean': mean_cv,
            'cv_std': std_cv,
            'test_acc': test_acc,
            'model': model,
            'time': elapsed_time
        }
        
        print(f"Done ({elapsed_time:.1f}s)")

        print(f"\n{name:<25} {mean_cv:.4f}          {std_cv:.4f}     {test_acc:.4f}          {elapsed_time:.1f}")
    print("-" * 85)
    
    #Finds best performing algorithm
    best_algo = max(results, key=lambda x: results[x]['cv_mean'])
    print(f"\nBest performing algorithm (by CV mean): {best_algo}")
    print(f"  CV Mean: {results[best_algo]['cv_mean']:.4f} ± {results[best_algo]['cv_std']:.4f}")
    print(f"  Test Accuracy: {results[best_algo]['test_acc']:.4f}")
    return results

#PART 3
def fast_feature_selection(model, X_train, y_train, X_test, y_test, feature_names, k_features=20):

    print(f"  Step 1: Reducing from {X_train.shape[1]} to top {k_features} features using ANOVA...")
    
    #Quick feature ranking using ANOVA
    selector = SelectKBest(f_classif, k=min(k_features, X_train.shape[1]))
    X_train_reduced = selector.fit_transform(X_train, y_train)
    
    #Get indices of selected features
    selected_mask = selector.get_support()
    selected_indices = [i for i, is_selected in enumerate(selected_mask) if is_selected]
    reduced_feature_names = [feature_names[i] for i in selected_indices]
    
    print(f"  Step 2: Running forward selection on {len(selected_indices)} features...")
    
    #Use reduced CV for feature selection (10x5 instead of 10x10)
    selection_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    
    #Create a new model instance for feature selection
    if isinstance(model, MLPClassifier):
        selection_model = MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=100,
            random_state=RANDOM_STATE,
            early_stopping=True,
            verbose=False
        )
    else:
        selection_model = model.__class__(**model.get_params())
    
    #Forward selection on reduced features
    sfs = SequentialFeatureSelector(
        estimator=selection_model,
        n_features_to_select='auto',
        direction='forward',
        scoring='accuracy',
        cv=selection_cv,
        n_jobs=-1
    )
    
    sfs.fit(X_train_reduced, y_train)
    final_mask = sfs.get_support()
    final_indices = [selected_indices[i] for i, is_selected in enumerate(final_mask) if is_selected]
    final_features = [feature_names[i] for i in final_indices]
    
    #Evaluate final feature set
    if final_indices:
        X_train_final = X_train[:, final_indices]
        X_test_final = X_test[:, final_indices]
        
        #Evaluation with 10x10 CV
        eval_cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train_final, y_train, cv=eval_cv, 
                                 scoring='accuracy', n_jobs=-1)
        
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_train_final, y_train)
        test_acc = model_copy.score(X_test_final, y_test)
        
        return scores.mean(), scores.std(), test_acc, final_features
    else:
        return None, None, None, []


def perform_feature_selection(results_part2, X_train, y_train, X_test, y_test, feature_names):

    print("\n" + "o" * 70)
    print("PART 3: Feature Selection Optimization")
    print("o" * 70)
    
    m = X_train.shape[1]
    print(f"\nNumber of features available: {m}")
    print("Note: This might take a few seconds\n")
    
    print("-" * 115)
    print(f"{'Algorithm':<25} {'Best Features Found':<45} {'CV Mean':<12} {'CV Std':<10} {'Test Acc':<10} {'Time (s)':<10}")
    print("-" * 115)
    
    feature_selection_results = {}
    
    for name, model_data in results_part2.items():
        print(f"\nProcessing: {name}")
        model = model_data['model']
        baseline_acc = model_data['cv_mean']
        
        start_time = time.time()
        
        if name == 'Neural Network':
            k_features = 15  
        elif name == 'KNN':
            k_features = 25
        else:
            k_features = 20
        
        cv_mean, cv_std, test_acc, feat_list = fast_feature_selection(
            model, X_train, y_train, X_test, y_test, feature_names, k_features
        )
        
        elapsed_time = time.time() - start_time
        
        if cv_mean is not None:
            print(f"  Selected {len(feat_list)} features (took {elapsed_time:.1f}s)")
            print(f"  Baseline CV: {baseline_acc:.4f} → New CV: {cv_mean:.4f} "
                  f"({'+' if cv_mean >= baseline_acc else ''}{cv_mean - baseline_acc:.4f})")
            
            feature_selection_results[name] = {
                'best_features': feat_list,
                'num_features': len(feat_list),
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'test_acc': test_acc,
                'baseline_acc': baseline_acc,
                'improvement': cv_mean - baseline_acc
            }
            
            #Prints table row
            feat_display = str(feat_list[:5])
            if len(feat_list) > 5:
                feat_display = feat_display[:-1] + f", +{len(feat_list)-5} more]"
            
            print(f"{name:<25} {feat_display:<45} {cv_mean:.4f}       {cv_std:.4f}    {test_acc:.4f}       {elapsed_time:.1f}")
        else:
            print(f"  Feature selection failed, using all features")
            feature_selection_results[name] = {
                'best_features': feature_names[:10],  # Show first 10 as example
                'num_features': len(feature_names),
                'cv_mean': results_part2[name]['cv_mean'],
                'cv_std': results_part2[name]['cv_std'],
                'test_acc': results_part2[name]['test_acc'],
                'baseline_acc': results_part2[name]['cv_mean'],
                'improvement': 0
            }
            
            print(f"{name:<25} All 51 features{' ':>38} {results_part2[name]['cv_mean']:.4f}       "
                  f"{results_part2[name]['cv_std']:.4f}    {results_part2[name]['test_acc']:.4f}       {elapsed_time:.1f}")
    
    print("-" * 115)
    
    #Summary of improvements
    print("\n" + "o" * 75)
    print("Feature Selection Impact Summary")
    print("o" * 75)
    print("\n")
    print(f"{'Algorithm':<25} {'Features Used':<15} {'Baseline':<12} {'After FS':<12} {'Change':<10}")
    print("-" * 75)
    
    for name, results in feature_selection_results.items():
        change_str = f"+{results['improvement']:.4f}" if results['improvement'] > 0 else f"{results['improvement']:.4f}"
        print(f"{name:<25} {results['num_features']:<15} {results['baseline_acc']:.4f}      "
              f"{results['cv_mean']:.4f}      {change_str}")
    return feature_selection_results

def main():
    
    #Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler = load_and_preprocess()
    
    #Convert to numpy arrays so it's faster
    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values
    y_test_np = y_test.values
    
    #Part 2: Evaluate classifiers
    results_part2 = evaluate_classifiers(X_train_np, y_train_np, X_test_np, y_test_np)
    
    #Part 3: Feature selection
    results_part3 = perform_feature_selection(
        results_part2, X_train_np, y_train_np, X_test_np, y_test_np, feature_names
    )    
    return results_part2, results_part3

#Runs main
if __name__ == "__main__":
    results_part2, results_part3 = main()