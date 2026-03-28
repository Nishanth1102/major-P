import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('results/round_metrics.csv')

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(df['round'], df['accuracy'] * 100, marker='o', linewidth=2.5,
        markersize=8, color='steelblue', markerfacecolor='white',
        markeredgewidth=2, label='Global Accuracy')
ax.fill_between(df['round'], df['accuracy'] * 100, alpha=0.12, color='steelblue')
ax.set_xlabel('FL Round', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Federated Learning — Global Accuracy per Round\n(CIC-IDS 2018 | FedAvg | 2 Clients | Non-IID)', fontsize=13)
ax.set_ylim(50, 100)
ax.set_xticks(df['round'])
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=11)

for x, y in zip(df['round'], df['accuracy'] * 100):
    ax.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=9, color='steelblue')

plt.tight_layout()
plt.savefig('results/accuracy_plot.png', dpi=150)
print('Plot saved to results/accuracy_plot.png')