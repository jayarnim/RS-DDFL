import matplotlib.pyplot as plt


def score_plot(
    history: dict,
    metric: str,
    figsize: tuple=(8,5),
):
    plt.figure(figsize=figsize)
    plt.plot(history['loo'], label='LOO')
    plt.xlabel('EPOCH')
    plt.ylabel(metric)
    plt.title('LEAVE ONE OUT SCORE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def loss_plot(
    history: dict,
    loss: str,
    figsize: tuple=(8,5),
):
    plt.figure(figsize=figsize)
    plt.plot(history['trn'], label='TRN')
    plt.plot(history['val'], label='VAL')
    plt.xlabel('EPOCH')
    plt.ylabel(loss)
    plt.title('TRN vs. VAL')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
