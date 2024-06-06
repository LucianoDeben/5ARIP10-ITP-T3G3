import matplotlib.pyplot as plt


def plot_drr_enhancement(
    drr_body, drr_vessels, enhancement_factors, cmap="gray", vmax=20
):
    # Move tensors to CPU
    drr_body = drr_body.cpu()
    drr_vessels = drr_vessels.cpu()

    fig, axs = plt.subplots(1, len(enhancement_factors), figsize=(12, 4))

    for i, ef in enumerate(enhancement_factors):
        drr_combined = drr_body + ef * drr_vessels
        axs[i].imshow(drr_combined.squeeze(), cmap=cmap, vmax=vmax)
        axs[i].set_title(ef)
        axs[i].axis("off")
    plt.show()


def plot_results(
    drr_combined_low_enhancement, drr_combined_target, prediction, vmax=25
):

    # Move tensors to CPU
    drr_combined_low_enhancement = drr_combined_low_enhancement.cpu()
    drr_combined_target = drr_combined_target.cpu()
    prediction = prediction.cpu()

    # Create a figure and a set of subplots
    _, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot each image in a subplot
    axs[0].imshow(
        drr_combined_low_enhancement.squeeze().numpy(), cmap="gray", vmax=vmax
    )
    axs[0].set_title("DRR")

    axs[2].imshow(drr_combined_target.squeeze().numpy(), cmap="gray", vmax=vmax)
    axs[2].set_title("enhanced target")

    axs[1].imshow(prediction.detach().numpy().squeeze(), cmap="gray", vmax=vmax)
    axs[1].set_title("AI Enhanced")

    # Hide the axes labels
    for ax in axs:
        ax.axis("off")

    plt.show()
