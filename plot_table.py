import matplotlib.pyplot as plt
from platform import python_version as pythonversion
from matplotlib import __version__ as matplotlibversion

def draw(row_labels,table_vals):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    col_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    # row_labels = ['0', '1', '2','3']
    # table_vals = [[11, 12, 13,1], [21, 22, 23,2], [31, 32, 33,3],[55,77,66,4]]

    # Draw table
    the_table = plt.table(cellText=table_vals,
                        colWidths=[0.1] * 4,
                        rowLabels=row_labels,
                        colLabels=col_labels,
                        loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(4, 4)

    # Removing ticks and spines enables you to get the figure only with table
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.savefig('matplotlib-table.png', bbox_inches='tight', pad_inches=0.05)