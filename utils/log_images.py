import numpy as np
import matplotlib.pyplot as plt
IMG_LOG_DIR = 'data/runs/images'

def save_imgs(dict, args, epoch, max_images=0):
    input_images = dict['INPUTS']
    labels = dict['LABELS']
    #latents = dict['LATENTS']
    mus = dict['MUS']
    logvars = dict['LOGVARS']
    reconstructed_images = dict['RECS']

    n_images = input_images.size(0)

    for i in range(n_images):
      # save input images
      input_image = input_images[i].cpu().detach().numpy()
      input_image = np.transpose(input_image, (1, 2, 0))
      if input_image.shape[-1] == 1:
        input_image = np.squeeze(input_image,-1)
      #input_image = np.clip(input_image, 0, 1)
      open_cv_image = np.uint8(input_image * 255)
      fig, ax = plt.subplots()
      ax.imshow(input_image)
      fig.savefig(f"{IMG_LOG_DIR}/epoch{epoch}_image_{i}-0 (input).png")
      plt.close(fig)

      recon_image = reconstructed_images[i].cpu().detach().numpy()
      recon_image = np.transpose(recon_image, (1, 2, 0))
      if recon_image.shape[-1] == 1:
        recon_image = np.squeeze(recon_image,-1)
      recon_image = np.clip(recon_image, 0, 1)
      open_cv_image = np.uint8(recon_image * 255)
      fig, ax = plt.subplots()
      ax.imshow(open_cv_image)
      fig.savefig(f"{IMG_LOG_DIR}/epoch{epoch}_image_{i}-1 (reconstructed).png")
      plt.close(fig)

      if i >= max_images and max_images > 0:
        break