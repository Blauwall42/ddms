import matplotlib.pyplot as plt
import pandas as pd
import morris


def plot_results():
    df: pd.DataFrame = pd.read_csv('morris.csv', index_col=0)
    print(df)

    fig, ax = plt.subplots()
    ax.scatter(df['mu'], df['sigma'])
    for i, txt in enumerate(df.index):
        ax.annotate(txt, (df['mu'][i], df['sigma'][i]), (df['mu'][i]-3.5, df['sigma'][i]-3))
    ax.set_xlabel('μ')
    ax.set_ylabel('σ')
    plt.savefig('mu-sigma.svg', format='svg')

    fig, ax = plt.subplots()
    ax.scatter(df['mu_star'], df['sigma'])
    for i, txt in enumerate(df.index):
        ax.annotate(txt, (df['mu_star'][i], df['sigma'][i]), (df['mu_star'][i]-3.5, df['sigma'][i]-3))
    ax.set_xlabel('μ*')
    ax.set_ylabel('σ')
    plt.savefig('mu_star-sigma.svg', format='svg')

if __name__ == '__main__':
    morris.main()
    plot_results()
