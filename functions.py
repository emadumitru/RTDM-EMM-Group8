from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from scipy.io import arff
import warnings
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import Orange
from scipy.stats import ranksums
from Orange.data import Table, Domain, ContinuousVariable
from Orange.classification import CN2Learner
from mlxtend.frequent_patterns import apriori, association_rules
import random
from sklearn.tree import DecisionTreeRegressor
import multiprocessing
import pandas as pd
import Orange
from Orange.classification import CN2Learner
from Orange.data import Table, Domain, ContinuousVariable
from scipy.stats import ranksums, chi2_contingency
import numpy as np


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# folder_path = 'datasets'

def preprocess_dataset(df):
    for column in list(df.columns):
        if str(df[column][1]).startswith('b'):
            unique = df[column].unique()
            df[column] = df[column].map({unique[0]: 0, unique[1]: 1})

        if df[column].isna().sum()/len(df) > 0.80:
            df = df.drop([column])
    return df

def fill_na_with_mode(df):
    for col in df.columns:
        mode_val = df[col].mode().iloc[0]
        df[col].fillna(mode_val, inplace=True)
    return df

def target_type(df):
    df = df.dropna()
    last = df.columns[-1]
    target = df[last]
    diff = target.unique()
    if len(diff) <= 2:
        return "Binary"
    elif len(diff) > 2:
        return "Multinomial"
    return 0

def infer_type(value) -> str:
    if isinstance(value, str):
        if value.lower() in ['true', 'false']:
            return 'bool'
        try:
            int(value)
            return 'int64'
        except ValueError:
            try:
                float(value)
                return 'float64'
            except ValueError:
                pass
    return 'string'

def get_object_column_types(df: pd.DataFrame) -> dict:
    column_types = {}
    
    for column in df.select_dtypes(include='object').columns:
        most_common_type = Counter(df[column].map(infer_type)).most_common(1)[0][0]

        if most_common_type == 'string':
            unique_values_ratio = df[column].nunique() / len(df)
            if unique_values_ratio <= 0.2 or (df[column].nunique() <= 3 and len(df) < 20):
                most_common_type = "categorical"
        
        column_types[column] = most_common_type

    return column_types

def get_datasets_characteristics(file, in_df: str, df: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    csv = in_df
    dtypes = csv.dtypes.unique()

    object_column_type_dict = get_object_column_types(csv)

    df.loc[file, 'crow'] = len(csv)
    df.loc[file, 'ccol'] = len(csv.columns)
    for dtype in dtypes:
        num_cols = len(csv.select_dtypes(include=[dtype]).columns)
        df.loc[file, str(dtype)] = num_cols
    df.loc[file, 'cnan'] = csv.isnull().sum().sum()

    for item in set(object_column_type_dict.values()):
        if item == str:
            item2 = 'string'
            df.loc[file, item2] = list(object_column_type_dict.values()).count(item)
        elif item == 'int64' or item == 'float64':
            df.loc[file, item] += list(object_column_type_dict.values()).count(item)
        else:
            df.loc[file, item] = list(object_column_type_dict.values()).count(item)

    return df.fillna(0)

def load_datasets(path):
    metadata_df = pd.DataFrame()
    num_cols = []
    num_rows = []
    datasets = {}
    names = [] 
    t_type = []
    dataset_names = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.arff')]
    new_charac = pd.DataFrame(index=dataset_names)
    for i in range(len(dataset_names)):
        try:
            data = arff.loadarff(path+'/'+dataset_names[i])
            df = pd.DataFrame(data[0])
            df = preprocess_dataset(df)
            name_df = dataset_names[i].replace(".arff", "")
            names.append(name_df)
            num_cols.append(df.shape[1])
            num_rows.append(df.shape[0])
            t_type.append(target_type(df))
            new_charac = get_datasets_characteristics(dataset_names[i], df, new_charac)
            df = fill_na_with_mode(df)
            datasets[name_df] = df
        except:
            pass
    metadata_df["Datasets"] = names
    metadata_df["num_columns"] = num_cols
    metadata_df["num_rows"] = num_rows
    metadata_df["target_type"] = t_type
    new_charac = new_charac[~(new_charac == 0).all(axis=1)]
    metadata_df['num_nan'] = new_charac['cnan'].values
    metadata_df['num_float'] = new_charac['float64'].values.astype(int)
    metadata_df['num_int'] = new_charac['int64'].values.astype(int)
    metadata_df['perc_nan'] = (metadata_df['num_nan'] / (metadata_df['num_rows'] * metadata_df['num_columns'])).round(5)*100
    return datasets, metadata_df

def sd_sd(data, target_column):
    """
    Subgroup Discovery (SD) using a decision tree.

    Parameters:
    - data: pd.DataFrame, the input data.
    - target_column: str, the name of the target column.

    Returns:
    - pd.DataFrame containing rules (as interpretable conditions) and their quality measures.
    """

    # Train a shallow decision tree
    tree = DecisionTreeRegressor(max_depth=2)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    tree.fit(X, y)

    # Extract rules from the tree
    rules = []
    path = tree.decision_path(X).todense()

    for sample in range(path.shape[0]):
        rule = []
        for node in range(path.shape[1]):
            if path[sample, node] == 0:
                continue
            if (tree.tree_.children_left[node] == tree.tree_.children_right[node]):
                continue
            if X.iloc[sample, tree.tree_.feature[node]] <= tree.tree_.threshold[node]:
                rule.append(f"{X.columns[tree.tree_.feature[node]]} <= {tree.tree_.threshold[node]:.2f}")
            else:
                rule.append(f"{X.columns[tree.tree_.feature[node]]} > {tree.tree_.threshold[node]:.2f}")
        rules.append(' AND '.join(rule))

    # Calculate quality for each rule (difference from global mean)
    overall_mean = y.mean()
    # qualities = [y.iloc[i] - overall_mean for i in range(len(y))]

    coverage_list = []
    quality_list = []
    support_list = []
    wracc_list = []
    confidence_list = []
    significance_list = []

    rules = list(dict.fromkeys(rules))

    for i in range(len(rules)):
        rule = rules[i]
        # quality = rule.quality
        conditions = rule.split(' AND ')
        # Apply each condition to filter the subgroup
        subgroup = data
        expressions = [" > ", " < ", " >= ", " <=", " == ", " != "]
        for condition in conditions:
            for ex in expressions:
                if ex in condition:
                    col_name, value = condition.split(ex)
                    if ex in [" > ",  " >= "]:
                        value = 1
                    elif ex in [" < ",  " <= "]:
                        value = 0
                    subgroup = subgroup[subgroup[col_name] == float(value)]
            # print(condition)

        subgroup_mean = subgroup[target_column].mean()
        coverage = len(subgroup) / len(data)
        # print("len sub: "+ str(len(subgroup)))

        TP = 0
        for i in list(subgroup.index)[1:]:
            if (subgroup[target_column][i] == 1) & (data[target_column][i] == 1):
                TP += 1
        FP = len(subgroup) - TP

        support = TP / len(data)
        wracc = TP / len(data) - ((TP + FP) / len(data)) * (len(data[data[target_column] == 1]) / len(data))
        quality = subgroup_mean - overall_mean
        non_subgroup_data = data[~data.index.isin(subgroup.index)].dropna()
        subgroup = subgroup.dropna()
        _, significance = ranksums(subgroup[target_column], non_subgroup_data[target_column])
       
        # Calculate Confidence
        confidence = (TP / len(subgroup)) if len(subgroup) > 0 else 0

        coverage_list.append(coverage)
        quality_list.append(quality)
        support_list.append(support)
        wracc_list.append(wracc)
        confidence_list.append(confidence)
        significance_list.append(significance)

    # Number of subgroups
    num_subgroups = len(rules)

    # Average length of subgroups
    avg_length_subgroups = np.mean([len(rule.split(" AND ")) for rule in rules])

    # Convert rules and metrics into DataFrame
    rules_df = pd.DataFrame({
        'rule': rules,
        'quality': quality_list,
        'coverage': coverage_list,
        'support': support_list,

    })

    result_metrics = {
        'Average Quality': np.mean(rules_df.quality),
        'Average Coverage': np.mean(coverage_list),
        'Average Support': np.mean(support_list),
        'Average WRAcc': np.mean(wracc_list),
        'Average Significance': np.mean(significance_list),
        'Average Confidence': np.mean(confidence_list),
        'Number of Subgroups': num_subgroups,
        'Average Length of Subgroups': avg_length_subgroups,
    }

    # print(result_metrics)
    return result_metrics

def cn2_sd_sd(data, target_column):
    """
    Subgroup Discovery using CN2 algorithm with adjusted parameters.

    Parameters:
    - data: pd.DataFrame, the input data.
    - target_column: str, the name of the target column.

    Returns:
    - pd.DataFrame containing rules (as interpretable conditions) and their quality measures.
    """

    # Convert all columns to string type for categorical interpretation
    data_str = data.astype(str)

    def get_subgroup(rule, data):
        condition_str = str(rule).split('IF ')[1].split(' THEN')[0].strip()
        conditions = condition_str.split(' AND ')

        # Apply each condition to filter the subgroup
        subgroup = data
        for condition in conditions:
            # Extract the column name, comparison operator, and value
            if "==" in condition:
                col_name, value = condition.split('==')
                subgroup = subgroup[subgroup[col_name] == float(value)]
            elif "!=" in condition:
                col_name, value = condition.split('!=')
                subgroup = subgroup[subgroup[col_name] == float(value)]
        return subgroup

    # Create Orange domain with explicit categories for each column
    domain_vars = []
    for col in data_str.columns:
        unique_vals = data_str[col].unique()
        domain_vars.append(Orange.data.DiscreteVariable.make(col, values=unique_vals))
    domain = Domain(domain_vars[:-1], domain_vars[-1])

    table = Table.from_list(domain, data_str.values)

    # Create learner with adjusted parameters and induce rules
    learner = CN2Learner()
    learner.rule_finder.search_algorithm.beam_width = 10
    learner.rule_finder.general_validator.min_covered_examples = 15
    classifier = learner(table)

    rules = []
    for rule in classifier.rule_list:
        # Using string representation to extract the rule conditions
        rule_str = str(rule).split("->")[0].strip()
        subgroup = get_subgroup(rule, data)
        coverage = len(subgroup) / len(data)

        TP = 0
        for i in list(subgroup.index)[1:]:
            if (subgroup[target_column][i] == 1) & (data[target_column][i] == 1):
                TP += 1

        # true_positives = sum([data[target_column][idx] == 1 for idx in range(len(coverage))])
        FP = len(subgroup) - TP
        # true_negatives = len(data) - len(subgroup) - FP

        support = TP / len(data)
        quality = rule.quality

        # wracc = (len(subgroup) / len(data)) * (((true_positives + true_negatives) / len(subgroup)) - (len(data[data[target_column] == 1]) / len(data)))
        wracc = TP / len(data) - ((TP + FP) / len(data)) * (len(data[data[target_column] == 1]) / len(data))


        # Calculate Significance using likelihood ratio statistic
        non_subgroup_data = data[~data.index.isin(subgroup.index)].dropna()
        subgroup = subgroup.dropna()
        # subgroup = subgroup[target_column].astype(int)
        # non_subgroup_data = non_subgroup_data[target_column].astype(int)
        # print(subgroup[target_column])
        # print(non_subgroup_data.iloc[:,0])
        _, significance = ranksums(subgroup[target_column], non_subgroup_data.iloc[:,0])
        # significance = 2 * true_positives * np.log(true_positives / (len(data[target_column]) * (len(subgroup)/len(data))))

        # Calculate Confidence
        confidence = (TP / len(subgroup)) if len(subgroup) > 0 else 0

        rules.append((rule_str, quality, coverage, support, wracc, significance, confidence))

    # Convert rules into DataFrame
    rules_df = pd.DataFrame(rules, columns=['rule', 'quality', 'coverage', 'support', 'WRAcc', 'Significance', 'Confidence'])

    num_subgroups = len(rules)
    av_len_subgroups = sum(len(rule[0].split(' AND ')) for rule in rules) / num_subgroups

    result_metrics = {
        'Average Quality': np.mean(rules_df.quality),
        'Average Coverage': np.mean(rules_df.coverage),
        'Average Support': np.mean(rules_df.support),
        'Average WRAcc': np.mean(rules_df.WRAcc),
        'Average Significance': np.mean(rules_df.Significance),
        'Average Confidence': np.mean(rules_df.Confidence),
        'Number of Subgroups': num_subgroups,
        'Average Length of Subgroups': av_len_subgroups,
    } 

    # return rules_df.sort_values(by='quality', ascending=False), num_subgroups, len_subgroups
    return result_metrics

def sd_map_sd(data, target_column, min_support):
    """
    SD-Map algorithm for subgroup discovery with binarized input.

    Parameters:
    - data: pd.DataFrame, the input data.
    - target_column: str, the name of the target column.
    - min_support: float, minimum support for the Apriori algorithm.

    Returns:
    - pd.DataFrame containing subgroups and their quality measures.
    """

    # Binarize the input data based on median values
    for column in data.columns:
        if column != target_column:
            median_value = data[column].median()
            data[column] = (data[column] > median_value).astype(int)

    # Drop the target column and compute frequent itemsets using Apriori
    frequent_itemsets = apriori(data.drop(columns=[target_column]), min_support=min_support, use_colnames=True)

    # Compute the quality of each subgroup
    overall_mean = data[target_column].mean()
    quality_measures = []
    coverage_list = []
    support_list = []
    wracc_list = []
    significance_list = []
    confidence_list = []
    len_list = []

    for _, row in frequent_itemsets.iterrows():
        subgroup_data = data[np.logical_and.reduce([data[col] for col in row['itemsets']])]
        # print(subgroup_data)
        subgroup_mean = subgroup_data[target_column].mean()
        coverage = len(subgroup_data) / len(data)
        len_rule = len(row['itemsets'])
        
        
        # Calculate Wilcoxon Rank Sum Test (Significance)
        non_subgroup_data = data[~np.logical_and.reduce([data[col] for col in row['itemsets']])]
        _, p_value = ranksums(subgroup_data[target_column], non_subgroup_data[target_column])
        
        # Calculate TP, TN, FP, FN for the subgroup
        TP = len(subgroup_data[subgroup_data[target_column] == 1])
        FP = len(subgroup_data[subgroup_data[target_column] == 0])

        support = TP / len(data)

        # Calculate Confidence
        confidence = (TP / len(subgroup_data)) if len(subgroup_data) > 0 else 0

        # Calculate WRAcc
        wracc = TP / len(data) - ((TP + FP) / len(data)) * (len(data[data[target_column] == 1]) / len(data))

        quality_measures.append(subgroup_mean - overall_mean)
        coverage_list.append(coverage)
        support_list.append(support)
        wracc_list.append(wracc)
        significance_list.append(p_value)
        confidence_list.append(confidence)
        len_list.append(len_rule)

    frequent_itemsets['quality'] = quality_measures
    frequent_itemsets['WRacc'] = wracc_list
    frequent_itemsets['Significance'] = significance_list
    frequent_itemsets['Confidence'] = confidence_list

    # Rank subgroups based on quality
    ranked_subgroups = frequent_itemsets.sort_values(by='quality', ascending=False)

    result_metrics = {
        'Average Quality': np.mean(quality_measures),
        'Average Coverage': np.mean(coverage_list),
        'Average Support': np.mean(support_list),
        'Average WRacc': np.mean(wracc_list),
        'Average Significance': np.mean(significance_list),
        'Average Confidence': np.mean(confidence_list),
        'Number of Subgroups': len(frequent_itemsets),
        'Average Length of Subgroups': np.mean(len_list),
    }

    return result_metrics

def dssd_sd(data, target_column, min_support):
    """
    Direct Subgroup Set Discovery (DSSD) algorithm for subgroup discovery with binarized input.

    Parameters:
    - data: pd.DataFrame, the input data.
    - target_column: str, the name of the target column.
    - min_support: float, minimum support for the Apriori algorithm.

    Returns:
    - pd.DataFrame containing non-redundant subgroups and their quality measures.
    """

    # Binarize the input data based on median values
    for column in data.columns:
        if column != target_column:
            median_value = data[column].median()
            data[column] = (data[column] > median_value).astype(int)

    # Compute frequent itemsets using Apriori
    frequent_itemsets = apriori(data.drop(columns=[target_column]), min_support=min_support, use_colnames=True)

    # Compute the quality of each subgroup
    overall_mean = data[target_column].mean()

    quality_measures = []
    coverage_list = []
    support_list = []
    wracc_list = []
    significance_list = []
    confidence_list = []

    # print(frequent_itemsets["itemsets"])

    for _, row in frequent_itemsets.iterrows():
        subgroup_data = data[np.logical_and.reduce([data[col] for col in row['itemsets']])]

        subgroup_mean = subgroup_data[target_column].mean()
        quality = subgroup_mean - overall_mean

        # Calculate coverage
        coverage = len(subgroup_data) / len(data)

        # Calculate Significance
        non_subgroup_data = data[~data.index.isin(subgroup_data.index)]
        _, p_value = ranksums(subgroup_data[target_column], non_subgroup_data[target_column])

        TP = len(subgroup_data[subgroup_data[target_column] == 1])
        FP = len(subgroup_data[subgroup_data[target_column] == 0])

        # Calculate Confidence
        confidence = (TP / len(subgroup_data)) if len(subgroup_data) > 0 else 0

        # Calculate WRAcc
        wracc = TP / len(data) - ((TP + FP) / len(data)) * (len(data[data[target_column] == 1]) / len(data))
        
        support = TP / len(data)

        quality_measures.append(subgroup_mean - overall_mean)
        coverage_list.append(coverage)
        support_list.append(support)
        wracc_list.append(wracc)
        significance_list.append(p_value)
        confidence_list.append(confidence)

    frequent_itemsets['quality'] = quality_measures
    frequent_itemsets['coverage'] = coverage_list
    frequent_itemsets['support'] = support_list
    frequent_itemsets['WRAcc'] = wracc_list
    frequent_itemsets['Significance'] = significance_list
    frequent_itemsets['Confidence'] = confidence_list

    # Sort subgroups based on quality
    sorted_subgroups = frequent_itemsets.sort_values(by='quality', ascending=False)

    # Prune redundant subgroups to get a set of non-redundant subgroups
    non_redundant_subgroups = []
    for _, row in sorted_subgroups.iterrows():
        is_redundant = False
        for nr_subgroup in non_redundant_subgroups:
            if row['itemsets'].issubset(nr_subgroup):
                is_redundant = True
                break
        if not is_redundant:
            non_redundant_subgroups.append(row['itemsets'])

    non_redundant_subgroups_df = sorted_subgroups[sorted_subgroups['itemsets'].isin(non_redundant_subgroups)]

    result_metrics = {
        'Average Quality': np.mean(non_redundant_subgroups_df.quality),
        'Average Coverage': np.mean(non_redundant_subgroups_df.coverage),
        'Average Support': np.mean(non_redundant_subgroups_df.support),
        'Average WRAcc': np.mean(non_redundant_subgroups_df.WRAcc),
        'Average Significance': np.mean(non_redundant_subgroups_df.Significance),
        'Average Confidence': np.mean(non_redundant_subgroups_df.Confidence),
        'Number of Subgroups': len(non_redundant_subgroups_df["quality"]),
        'Average Length of Subgroups': np.mean([len(rule) for rule in frequent_itemsets["itemsets"]]),
    }

    return result_metrics

def nmeef_sd2(data, target_column, n_generations=10, population_size=100):
    def crossover(parent1, parent2):
        """One-point crossover."""
        point = random.randint(0, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

    def mutate(rule):
        """Bit-flip mutation."""
        index = random.randint(0, len(rule) - 1)
        new_bit = '1' if rule[index] == '0' else '0'
        return rule[:index] + new_bit + rule[index + 1:]

    def initialize_population():
        """Initialize a diverse random population of binary-coded rules."""
        return ["".join(random.choice(['0', '1']) for _ in range(data.shape[1] - 1))
                for _ in range(population_size)]

    def evaluate_rule(rule):
        """Evaluate the quality of a rule based on *metrics*."""
        try:
            subgroup_data = data[np.logical_and.reduce(
                [(data[columns[idx]] == 1) if bit == '1' else (data[columns[idx]] == 0) for idx, bit in enumerate(rule)])]
            if len(subgroup_data) == 0:
                return (0, 0, 0, 0, 0, 0, rule)
            subgroup_mean = subgroup_data[target_column].mean()
            non_subgroup_data = data[~data.index.isin(subgroup_data.index)]
            _, significance = ranksums(subgroup_data[target_column], non_subgroup_data[target_column])
            TP = len(subgroup_data[subgroup_data[target_column] == 1])
            FP = len(subgroup_data[subgroup_data[target_column] == 0])
            confidence = (TP / len(subgroup_data)) if len(subgroup_data) > 0 else 0
            wracc = TP / len(data) - ((TP + FP) / len(data)) * (len(data[data[target_column] == 1]) / len(data))
            coverage = len(subgroup_data) / len(data)
            support = TP / len(data)
            quality = (subgroup_mean - overall_mean)
            return (quality, coverage, support, wracc, significance, confidence, rule)
        except:
            return (0, 0, 0, 0, 0, 0, rule)

    columns = list(data.drop(columns=[target_column]).columns)
    overall_mean = data[target_column].mean()
    population = initialize_population()
    best_rules = []
    best_metrics = pd.DataFrame(columns=["quality", "coverage", "support", "WRAcc", "significance", "confidence"])

    for _ in range(n_generations):
        rule_metrics = []
        for rule in population:
            temp = evaluate_rule(rule)
            if type(temp) == tuple:
                rule_metrics.append(temp)
        rule_metrics = pd.DataFrame(rule_metrics, columns=["quality", "coverage", "support", "WRAcc", "significance", "confidence", "rule"])
        sorted_population = list(rule_metrics.sort_values("quality", ascending=False)["rule"])
        best_rules.extend(sorted_population[:5])
        top_5_metrics = rule_metrics.sort_values("quality", ascending=False)[["quality", "coverage", "support", "WRAcc", "significance", "confidence"]][:5]
        best_metrics = pd.concat([best_metrics, top_5_metrics], ignore_index=True)
        parents = sorted_population[:population_size // 2]
        offspring = []
        for i in range(0, len(parents) // 2 - 2, 2):
            offspring1, offspring2 = crossover(parents[i], parents[i + 1])
            offspring.append(mutate(offspring1))
            offspring.append(mutate(offspring2))
        population = parents + offspring

    result_metrics = {
        'Average Quality': np.mean(best_metrics.quality),
        'Average Coverage': np.mean(best_metrics.coverage),
        'Average Support': np.mean(best_metrics.support),
        'WRAcc': np.mean(best_metrics.WRAcc),
        'Significance': np.mean(best_metrics.significance),
        'Confidence': np.mean(best_metrics.confidence),
        'Number of Subgroups': len(best_rules),
        'Average Length of Subgroups': np.mean([str(rule).count("1") for rule in best_rules]),
    }

    return result_metrics

def apriori_sd2(data, target_column, min_support=0.1, metric="lift", min_threshold=1):

    def calculate_wracc(antecedent, consequent, data):
        rule = data[list(antecedent)].all(axis=1)  # Convert antecedent to a rule
        subgroup_data = data[rule]
        N = len(data)
        TP = len(subgroup_data[subgroup_data[consequent] == 1])
        FP = len(subgroup_data[subgroup_data[consequent] == 0])
        WRAcc = TP / N - (TP + FP) / N * (len(data[data[consequent] == 1]) / len(data))
        return WRAcc
    
    def calculate_significance(antecedent_items, consequent_item, data):
        rule = data[antecedent_items].all(axis=1)
        subgroup_data = data[rule]
        non_subgroup_data = data[~rule]
        _, p_value = ranksums(subgroup_data[consequent_item], non_subgroup_data[consequent_item])
        return p_value
    
    def _get_default_metrics():
        return {
            'Average Quality': 0,
            'Average Coverage': 0,
            'Average Support': 0,
            'WRAcc': 0,
            'Significance': 0,
            'Confidence': 0,
            'Number of Subgroups': 0,
            'Average Length of Subgroups': 0,
        }

    def calculate_significance(antecedent_items, consequent_item, data):
        rule = data[antecedent_items].all(axis=1)
        subgroup_data = data[rule]
        non_subgroup_data = data[~rule]
        
        # Check variance
        if np.var(subgroup_data[consequent_item]) == 0 or np.var(non_subgroup_data[consequent_item]) == 0:
            return 0

        _, p_value = ranksums(subgroup_data[consequent_item], non_subgroup_data[consequent_item])
        return p_value

    
    for column in data.select_dtypes(['int64', 'float64']).columns:
        threshold = data[column].median()
        data[column] = (data[column] > threshold).astype(int)

    data_encoded = pd.get_dummies(data)

    frequent_itemsets = apriori(data_encoded, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return _get_default_metrics()

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    target_rules = rules[rules['consequents'] == frozenset({target_column})]
    
    if target_rules.empty:
        return _get_default_metrics()
    
    WRAcc_list = []
    significance_list = []
    data = data.dropna()
    for i in range(len(target_rules)):
        antecedent_items = list(target_rules.iloc[i]['antecedents'])
        consequent_item = list(target_rules.iloc[i]['consequents'])[0]
        wracc = calculate_wracc(antecedent_items, consequent_item, data)
        significance = calculate_significance(antecedent_items, consequent_item, data)
        WRAcc_list.append(wracc)
        significance_list.append(significance)

    target_rules = target_rules.sort_values(by=metric, ascending=False)

    target_rules['rule'] = target_rules['antecedents'].apply(lambda x: ' AND '.join(list(x)))
    target_rules['consequent'] = target_rules['consequents'].apply(lambda x: list(x)[0])
    target_rules['coverage'] = target_rules['support'] / target_rules['antecedent support']


    result_metrics = {
        'Average Quality': None,
        'Average Coverage': np.mean(target_rules.coverage),
        'Average Support': np.mean(target_rules.support),
        'WRAcc': np.mean(WRAcc_list),
        'Significance': np.mean(significance_list),
        'Confidence': np.mean(target_rules.confidence),
        'Number of Subgroups': len(target_rules),
        'Average Length of Subgroups': np.mean([len(rule) for rule in target_rules["antecedents"]]),
    }

    return result_metrics

def run_function_with_timeout(func, timeout, *args, **kwargs):
    with multiprocessing.Pool(processes=1) as pool:
        result = pool.apply_async(func, args=args, kwds=kwargs)
        try:
            return result.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            print(f"{func.__name__} timed out")
            return {}

def run_algorithms(df, name):
    target = df.columns[-1]
    results = {}
    
    # print("START Dataset: ", name)
    
    functions = [
        ("sd", sd_sd, {}),
        ("cn2_sd", cn2_sd_sd, {}),
        ("sd_map", sd_map_sd, {"min_support": 0.1}),
        ("dssd", dssd_sd, {"min_support": 0.1}),
        ("nnmeef", nmeef_sd2, {}),
        ("a", apriori_sd2, {"min_threshold": 0.1})
    ]

    for res_key, func, kwargs in functions:
        values = run_function_with_timeout(func, 180, df, target, **kwargs if kwargs else {}).values()
        results[res_key] = list(values)

    # print("END Dataset: ", name)
    return results

def get_metadata(path, save_path):
    datasets, metadata_df = load_datasets(path)

    meta_2 = {}
    for key in datasets:
        results = run_algorithms(datasets[key], key)
        meta_2[key] = results

    df = pd.DataFrame.from_dict(meta_2, orient='index')
    df = df.stack().apply(pd.Series).reset_index()
    df.columns = ['Datasets', 'Algorithm', 'Quality', 'Coverage', 'Support', 'WRAcc', "Significance", "Confidence", '# of Subgroups', 'Length of Rules']

    temp = pd.merge(metadata_df, df, on=["Datasets"])
    temp.to_csv(f"{save_path}/full_metadata.csv", index=False)

    return temp

def clean_metada(data, save_path):
    data['Length of Rules'] = data['Length of Rules'].fillna(0)
    data = data[data['Length of Rules'] != 0]    
    data.to_csv(f"{save_path}/clean_metadata.csv", index=False)
    return data

def get_mean_var(data):
    columns_to_process = ['Quality', 'Coverage', 'Support', 'WRAcc', 'Significance', 'Confidence', '# of Subgroups', 'Length of Rules']
    result = data.groupby('Algorithm')[columns_to_process].agg(['mean', 'var']).reset_index()
    return result

def bin_3(df, cols_yes):
    for col in df.columns:
        if col in cols_yes:
            bins = [-float('inf'), df[col].quantile(0.33), df[col].quantile(0.67), float('inf')]
            labels = ['Low', 'Medium', 'High']
            df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df

def bin_2(df, cols_yes):
    for col in df.columns:
        if col in cols_yes:
            bins = [-float('inf'), df[col].quantile(0.5), float('inf')]
            labels = ['Low','High']
            df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df

def cn2_meta(data, target_column, beam = 5, mincov = 3):
    data_str = data.astype(str)

    def get_subgroup(rule, data):
        condition_str = str(rule).split('IF ')[1].split(' THEN')[0].strip()
        conditions = condition_str.split(' AND ')

        subgroup = data
        for condition in conditions:
            if "==" in condition:
                col_name, value = condition.split('==')
                subgroup = subgroup[subgroup[col_name] == value]
            elif "!=" in condition:
                col_name, value = condition.split('!=')
                subgroup = subgroup[subgroup[col_name] == value]
        return subgroup

    domain_vars = []
    for col in data_str.columns:
        unique_vals = data_str[col].unique()
        domain_vars.append(Orange.data.DiscreteVariable.make(col, values=unique_vals))
    domain = Domain(domain_vars[:-1], domain_vars[-1])

    table = Table.from_list(domain, data_str.values)

    learner = CN2Learner()
    learner.rule_finder.search_algorithm.beam_width = beam
    learner.rule_finder.general_validator.min_covered_examples = mincov
    classifier = learner(table)

    rules = []
    for rule in classifier.rule_list:
        rule_str = str(rule).split("->")[0].strip()
        subgroup = get_subgroup(rule, data)
        coverage = len(subgroup) / len(data)

        if coverage == 0:
            continue    
        TP = 0
        for i in list(subgroup.index)[1:]:
            if (subgroup[target_column][i] == 1) & (data[target_column][i] == 1):
                TP += 1

        FP = len(subgroup) - TP

        support = TP / len(data)
        quality = rule.quality

        wracc = TP / len(data) - ((TP + FP) / len(data)) * (len(data[data[target_column] == 1]) / len(data))


        non_subgroup_data = data[~data.index.isin(subgroup.index)].dropna()
        subgroup = subgroup.dropna()
        contingency = pd.crosstab(data.index.isin(subgroup.index), data[target_column])
        chi2, significance, _, _ = chi2_contingency(contingency)
        # significance = 2 * TP * np.log(TP +1/ (len(data[target_column]) * (len(subgroup)+1/len(data))))

        confidence = (TP / len(subgroup)) if len(subgroup) > 0 else 0

        rules.append((rule_str, quality, coverage, support, wracc, significance, confidence))

    rules_df = pd.DataFrame(rules, columns=['rule', 'quality', 'coverage', 'support', 'WRAcc', 'Significance', 'Confidence'])

    num_subgroups = len(rules)
    av_len_subgroups = sum(len(rule[0].split(' AND ')) for rule in rules) / num_subgroups

    result_metrics = {
        'Average Quality': np.mean(rules_df.quality),
        'Average Coverage': np.mean(rules_df.coverage),
        'Average Support': np.mean(rules_df.support),
        'Average WRAcc': np.mean(rules_df.WRAcc),
        'Average Significance': np.mean(rules_df.Significance),
        'Average Confidence': np.mean(rules_df.Confidence),
        'Number of Subgroups': num_subgroups,
        'Average Length of Subgroups': av_len_subgroups,
    } 

    return result_metrics, rules_df

def run_meta(data, save_path):
    metrics = ['Quality', 'Coverage', 'Support', 'WRAcc',
       'Significance', 'Confidence', '# of Subgroups', 'Length of Rules']
    df_evals = pd.DataFrame()
    results = {}

    for metric in metrics:
        mini_data = data[['Algorithm', 'num_columns', 'num_rows', 'target_type', 'num_nan', 'num_float',
        'num_int', 'perc_nan', metric]].copy()
        mini_data = bin_3(mini_data, [metric])
        mini_data = mini_data.dropna()

        result_metrics, rules_df = cn2_meta(mini_data, metric)
        results[metric] = result_metrics
        df_evals = pd.concat([df_evals, rules_df], ignore_index=True)

    filtered_df_3 = df_evals[df_evals['rule'].str.contains('Algorithm')].reset_index(drop=True)
    filtered_df_3.to_csv(f"{save_path}/filtered_rules_3.csv", index=False)

    metrics = ['Quality', 'Coverage', 'Support', 'WRAcc',
        'Significance', 'Confidence', '# of Subgroups', 'Length of Rules']
    df_evals = pd.DataFrame()
    results = {}

    for metric in metrics:
        mini_data = data[['Algorithm', 'num_columns', 'num_rows', 'target_type', 'num_nan', 'num_float',
        'num_int', 'perc_nan', metric]].copy()
        mini_data = bin_2(mini_data, [metric])
        mini_data = mini_data.dropna()

        result_metrics, rules_df = cn2_meta(mini_data, metric)
        results[metric] = result_metrics
        df_evals = pd.concat([df_evals, rules_df], ignore_index=True)

    filtered_df_2 = df_evals[df_evals['rule'].str.contains('Algorithm')].reset_index(drop=True)
    filtered_df_2.to_csv(f"{save_path}/filtered_rules_3.csv", index=False)

    return filtered_df_3, filtered_df_2
