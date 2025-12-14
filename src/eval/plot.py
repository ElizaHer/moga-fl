import matplotlib.pyplot as plt
import seaborn as sns

# Chinese font config
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_curves(df, out_prefix):
    fig, ax = plt.subplots(figsize=(10,6), dpi=200)
    ax.plot(df['round'], df['accuracy'], marker='o', label='准确率')
    ax.set_xlabel('轮次', labelpad=10)
    ax.set_ylabel('准确率', labelpad=10)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.25)
    plt.savefig(out_prefix + '_accuracy.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,6), dpi=200)
    ax.plot(df['round'], df['comm_energy'], marker='o', label='通信能耗')
    ax.plot(df['round'], df['comp_energy'], marker='o', label='计算能耗')
    ax.set_xlabel('轮次', labelpad=10)
    ax.set_ylabel('能耗（近似）', labelpad=10)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.25)
    plt.savefig(out_prefix + '_energy.png', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,6), dpi=200)
    ax.plot(df['round'], df['jain_index'], marker='o', label='Jain 公平指数')
    ax.set_xlabel('轮次', labelpad=10)
    ax.set_ylabel('公平性', labelpad=10)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.25)
    plt.savefig(out_prefix + '_fairness.png', bbox_inches='tight')
    plt.close(fig)
