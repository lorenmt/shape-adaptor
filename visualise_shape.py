import matplotlib.pyplot as plt

# define your shape here
shape = [224, 224, 112, 112, 56, 56, 56, 28, 28, 28, 14, 14, 14, 7]

# define your saving path here
save_path = 'visuals/my_shape.png'

y = 0
for i in range(len(shape)):
    scale = shape[i] / shape[0]
    x = 0 - (10 * scale) / 2
    rectangle = plt.Rectangle((x, y), 10 * scale, 3, color=(0.70, 0.71, 0.71))
    y -= 4
    plt.gca().add_patch(rectangle)
plt.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('{}'.format(save_path), bbox_inches='tight', pad_inches=0)
plt.close()
