import matplotlib.pyplot as plt


def plot_drr_enhancement(
    drr_body, drr_vessels, enhancement_factors, cmap="gray", vmax=20
):
    """
    Plot the DRR enhancement for different enhancement factors

    Args:
        drr_body: The DRR of the body
        drr_vessels: The DRR of the vessels
        enhancement_factors: The enhancement factors
        cmap: The color map
        vmax: The maximum value for the color map

    Returns:
        None
    """
    # Move tensors to CPU
    drr_body = drr_body.cpu()
    drr_vessels = drr_vessels.cpu()

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, len(enhancement_factors), figsize=(12, 4))

    # Plot each image in a subplot
    for i, ef in enumerate(enhancement_factors):
        drr_combined = drr_body + ef * drr_vessels
        axs[i].imshow(drr_combined.squeeze(), cmap=cmap, vmax=vmax)
        axs[i].set_title(ef)
        axs[i].axis("off")
    plt.show()


def plot_results(
    drr_combined_low_enhancement,
    drr_combined_target,
    prediction,
    latent_representation,
    vmax=25,
):
    fig = plt.figure(figsize=(12, 4))
    titles = ["DRR", "AI Enhanced", "Enhanced Target", "Latent Representation"]
    images = [
        drr_combined_low_enhancement.squeeze(),
        prediction.detach().squeeze(),
        drr_combined_target.squeeze(),
        latent_representation.detach().squeeze(),
    ]

    for i, (img, title) in enumerate(zip(images, titles), 1):
        ax = fig.add_subplot(1, 4, i)
        ax.imshow(
            img.numpy(), cmap="gray", vmax=vmax if i != 4 else None
        )  # vmax not applied to latent representation
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    return fig
