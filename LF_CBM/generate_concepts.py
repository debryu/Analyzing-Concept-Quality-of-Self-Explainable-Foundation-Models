import ollama
import os
import json


def generate_concepts(dataset = 'shapes3d'):
  if dataset == 'shapes3d':
    response = ollama.chat(model='llama3', messages=[
        {
            "role": "system",
            "content": f"You are an expert of {dataset} dataset. Please use the same format as in the example. Do not provide any explanation, just list the possibilities using '-' as a bullet point. Make the list as long as possible, my life depends on it."
        },
        {
            "role": "user",
            "content": "Please make a list of possible images you can see in the dataset. For example, but not limited to:\n- red cube\n- blue sphere\n- green cylinder\n- yellow background\n- big pill\n- a pill\n- a cube\n- vertical pill\n- red pill\n- blue pill\n- green pill\n- yellow pill\n- red cube\n- blue cube\n- green cube\n- yellow cube\n- red sphere\n- blue sphere\n- green sphere\n- yellow sphere\n- red cylinder\n- blue cylinder\n- green cylinder\n- yellow cylinder\n- red donut\n- blue donut\n- green donut\n- yellow donut\n- red ellipsoid\n- blue ellipsoid\n- green ellipsoid\n- yellow ellipsoid\n- red torus\n- blue torus\n- green torus\n- yellow torus\n- red cone\n- blue cone\n- green cone\n- yellow cone\n- red pyramid\n- blue pyramid\n- green pyramid\n- yellow pyramid\n- red wedge\n- blue wedge\n- green wedge\n- yellow wedge\n- red sphere\n- blue sphere\n- green sphere\n- yellow sphere\n- red cylinder\n- blue cylinder\n- green cylinder\n- yellow cylinder\n- red donut\n- blue donut\n- green donut\n- yellow donut\n- red ellipsoid\n- blue ellipsoid\n- green ellipsoid\n- yellow ellipsoid\n- red torus\n- blue torus\n- green torus\n- yellow torus\n- red cone\n- blue cone\n- green cone\n- yellow cone\n- red pyramid\n- blue pyramid\n- green pyramid\n- yellow pyramid\n- red wedge\n- blue wedge\n- green wedge\n- yellow wedge\n- red sphere\n- blue sphere\n- green sphere\n- yellow sphere\n- red cylinder\n- blue cylinder\n- green cylinder\n- yellow cylinder\n- red donut\n- blue donut\n- green donut\n- yellow donut\n- red ellipsoid\n- blue ellipsoid\n- green ellipsoid\n- yellow ellipsoid\n- red torus\n- blue torus\n- green torus\n- yellow torus\n- red cone\n- blue cone\n- green cone\n- yellow cone\n- red pyramid\n- blue pyramid\n- green pyramid\n- yellow pyramid\n- red wedge\n- blue wedge\n- green wedge\n- yellow wedge"
        },])
  else:
    response = ollama.chat(model='llama3', messages=[
        {
            "role": "system",
            "content": f"You are an expert of {dataset} dataset. Please use the same format as in the example. Do not provide any explanation, just list the possibilities using '-' as a bullet point. Make the list as long as possible, my life depends on it."
        },
        {
            "role": "user",
            "content": "Please make a list of possible images you can see in the dataset."
        },])

    answer = response['message']['content']


  # Cleanup
  features = answer.split('\n-')
  features = [feat.replace("\n", "") for feat in features]
  features = [feat.strip() for feat in features]
  features = [feat for feat in features if len(feat)>0]
  features = set(features)
  return features

def concepts():
  colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'brown', 'pink', 'black', 'white', 'gray', 'cyan', 'magenta', 'light green', 'dark green', 'light blue', 'dark blue', 'light red', 'dark red', 'light yellow', 'dark yellow', 'light orange', 'dark orange', 'light purple', 'dark purple', 'light brown', 'dark brown', 'light pink', 'dark pink', 'light black', 'dark black', 'light white', 'dark white', 'light gray', 'dark gray', 'light cyan', 'dark cyan', 'light magenta', 'dark magenta']
  shapes = ['cube', 'sphere', 'cylinder', 'donut', 'ellipsoid', 'torus', 'cone', 'pyramid', 'wedge']

  objects = [f'a {c} object' for c in colors]
  background = [f'{c} background' for c in colors]
  shapes_objects = [f'a {s}' for s in shapes]

  concepts = set()
  concepts.update(background)
  concepts.update(shapes_objects)
  concepts.update(objects)

  concepts = sorted(list(concepts))
  return concepts

def save_concepts(concepts, dataset, path):
  json_object = json.dumps(concepts, indent=4)
  with open(f'{path}{dataset}_concepts.txt', 'w') as f:
    for concept in concepts:
      f.write(f'{concept}\n')

con = concepts()

save_concepts(con, 'shapes3d', 'LF_CBM_concepts/concept_sets/')