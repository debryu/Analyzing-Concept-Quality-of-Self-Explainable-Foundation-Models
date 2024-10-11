from matplotlib import pyplot as plt

def collect_concept_examples(img_list,batch, concept_ids, activated = True):
    images, label, attributes = batch
    for i,img in enumerate(images):
        match = True
        for c in concept_ids:
            if attributes[i,c] != activated:
                match = False
                break
        if match:
            img_list.append(img.permute(1,2,0)) #Pytorch uses CxHxW, matplotlib uses HxWxC
        if len(img_list) >= 25:
            break
    return img_list

def plot(img_list):
    fig, axs = plt.subplots(5, 5)
    for i, img in enumerate(img_list):
        x = i // 5
        y = i % 5
        axs[x][y].imshow(img)
        axs[x][y].axis('off')
    plt.show()

def plot_concept_examples(loader, concepts:list, args, activated=True):
  if args.dataset == 'shapes3d':
    c = 42
  elif args.dataset == 'celeba':
    c = 40
  else:
    raise ValueError(f"Unknown dataset {args.dataset}.")
  
  
  ex_list = []
  for batch in loader:
      collect_concept_examples(ex_list, batch, concepts, activated=True)
      if len(ex_list) >= 25:
          break
  
  plot(ex_list)