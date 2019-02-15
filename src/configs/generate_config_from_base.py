import ruamel.yaml

with open('base.yml', 'r', encoding='utf-8') as f:
    base = ruamel.yaml.safe_load(f)

models = ['Inception', 'ResNet', 'SENet', 'Vgg']
dirs = ['/home/d/pancreas/box_data', '/home/d/pancreas/resample115_box_data']

for model in models:
    for dir in dirs:
        base['model']['name'] = model
        base['dataset']['dir'] = dir
        with open('{}_{}.yml'.format(model, dir.split('/')[-1]), 'w', encoding='utf-8') as f:
            ruamel.yaml.dump(base, f)
