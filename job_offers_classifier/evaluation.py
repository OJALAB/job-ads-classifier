def get_discrete_dist(labels, size):
    dist = np.zeros(size)
    unique, counts = np.unique(labels, return_counts=True)
    for l, c in zip(unique, counts):
        dist[l] = c / len(labels)
    return dist

# Metoda licząca wybrane metryki dla danej predykcji
def evaluate(pred, target):
    num_labels = pred.shape[1]
    metric_dict = {
        "acc/recall@1": Accuracy(task="multiclass", num_classes=num_labels),
        "macro_acc": Accuracy(task="multiclass", num_classes=num_labels, average='macro'),
        #"cf_matrix_true": ConfusionMatrix(num_classes=num_labels, normalize='all'),
        #"cf_matrix_all": ConfusionMatrix(num_classes=num_labels, normalize='all')
    }

    for i in range(2, min(num_labels, 11)):
        metric_dict[f"recall@{i}"] = Recall(task="multiclass", num_classes=num_labels, top_k=i)

    metrics = MetricCollection(metric_dict)
    results = metrics(torch.tensor(pred), torch.tensor(target))
    results['log-loss'] = torch.nn.functional.cross_entropy(torch.tensor(pred), torch.tensor(target), reduction='mean')
    return results

# Metoda biorąca model hierarchiczny i dokonująca predykcji dla każdego poziomu i następnie porządkująca je w DataFrame
def evaluate_levels(model, y, pred, pred_map, levels, plot_dist=None):
    results = {}
    for level in levels:
        level_pred, level_map = model.predict_for_level_bottom_up(pred, pred_map, output_level=level)
        level_y = model.remap_labels_to_level(y, level_map, output_level=level)

        level_label = f"level {level}"
        results[level_label] = {k: float(v) for k, v in evaluate(level_pred, level_y).items()}

        true_dist = get_discrete_dist(level_y, level_pred.shape[1])
        prob_pred_dist = level_pred.sum(axis=0) / level_pred.shape[0]
        #prob_pred_dist *= 1 / prob_pred_dist.sum()
        single_pred_dist = get_discrete_dist(np.argmax(level_pred, axis=1), level_pred.shape[1])

        results[level_label]['mse_pred_dist'] = ((true_dist - prob_pred_dist)**2).mean()
        results[level_label]['mse_single_pred_dist'] = ((true_dist - single_pred_dist)**2).mean()

        if plot_dist is not None and level in plot_dist:
            plt.rcParams.update({
                "figure.figsize": (16, 8),
            })
            width = 0.3
            ind = np.array(list(level_map.keys()))
            plt.bar(ind - width, true_dist, width, label='true dist')
            plt.bar(ind, prob_pred_dist, width, label='pred.mean(axis=0) dist')
            plt.bar(ind + width, single_pred_dist, width, label='pred.argmax(axis=1) dist')
            plt.xticks(ind, list(level_map.values()))
            plt.legend(loc="upper left", prop={'size': 16})
            plt.show()

    return results
