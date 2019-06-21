'''
功能：show img
'''


def show_img(img, title):
    # use for no GUI
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(img)
    plt.savefig(title + '.jpg')
    return

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # plt.text(x=cen_x, y=cen_y, s=str(furniture_cid))
    ax.imshow(img)
    plt.show()
