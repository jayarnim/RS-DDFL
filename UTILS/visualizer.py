import matplotlib.pyplot as plt


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
