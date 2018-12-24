import matplotlib.pyplot as plt


def plot_mag_phase(x):
    plt.rcParams['figure.figsize'] = (26, 8)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.matshow(x[:, :, 0], aspect='auto', interpolation='none', origin='lower')
    ax1.set_title('Magnitude')

    ax2.matshow(x[:, :, 1], aspect='auto', interpolation='none', origin='lower')
    ax2.set_title('Phase')
